import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch import Tensor
import pytorch_lightning as pl

from ..dataset import MaestroDataset
from aligner.utils.constants import *
from aligner.model import AlignerModel, ModelConfig
from aligner.utils.constants import *

from typing import NamedTuple, Optional, Literal
import gc


class LitModelConfig(NamedTuple):
    data_dir: str
    learning_rate: float
    batch_size: int
    nm_penalty: Optional[float] = None
    num_dataloader_workers: int = 0
    alpha_cls: float = 1.0
    start_at: int = 0


class AlignerLitModel(pl.LightningModule):

    def __init__(self,
                 model_config: ModelConfig,
                 lit_model_config: LitModelConfig):
        super().__init__()
        self.model = AlignerModel.from_config(model_config)
        self.config = lit_model_config
        self.save_hyperparameters()
        self.train_dataset = MaestroDataset(self.config.data_dir,
                                            split='train',
                                            shuffle=True,
                                            start_at=self.config.start_at)
        self.val_dataset = MaestroDataset(self.config.data_dir,
                                          split='validation')


    @staticmethod
    def metric_NM(Y_hat: torch.Tensor) -> float:
        pred_indices = torch.argmax(Y_hat, dim=-1)
        diffs = pred_indices[:, 1:] - pred_indices[:, :-1]
        return torch.mean(torch.any(diffs < 0, dim=-1).float(), dim=-1)


    @staticmethod
    def metric_MAE(Y_hat: Tensor,
                   Y: Tensor,
                   reduction: Literal['mean', 'none'] = 'mean',
                   beta: float = 0.5
                   ) -> Tensor | float:
        assert Y_hat.shape == Y.shape
        max_n_events = Y.shape[-1]

        y = Y.argmax(dim=-1).unsqueeze(-1).float()
        x = torch.arange(max_n_events).view(1,1,-1).type_as(y)

        # Recall Beta = 0.5.
        # Since x and y are discrete, smoothing is only applied
        # when x exactly equals y.
        is_smooth = torch.abs(x - y) < beta
        dist = (0.5 * (x - y) ** 2 / beta) * is_smooth \
             + (torch.abs(x - y) - 0.5 * beta) * ~is_smooth

        loss = Y_hat * dist             # weighted L1
        if reduction == 'none':
            return loss
        if reduction == 'mean':
            loss = loss.sum(dim=-1)     # weighted sum over the event axis
            loss = loss.mean()          # average everything else
            return loss
        raise ValueError


    @staticmethod
    def metric_RMSE(Y_hat: Tensor,
                    Y: Tensor,
                    reduction: Literal['sqrt', 'none'] = 'sqrt',
                    eps: float = 1e-9
                    ) -> Tensor | float:

        assert Y_hat.shape == Y.shape
        max_n_events = Y.shape[-1]

        y = Y.argmax(dim=-1).unsqueeze(-1).float()
        x = torch.arange(max_n_events).view(1,1,-1).type_as(y)

        dist = (x - y) ** 2
        loss = Y_hat * dist             # weighted L2

        if reduction == 'none':
            return loss
        if reduction == 'sqrt':
            loss = loss.sum(dim=-1)     # weighted sum over the event axis
            loss = torch.sqrt(loss + eps)
            return loss.mean()          # mean over the frame & batch axes
        raise ValueError


    @staticmethod
    def metric_CE(Y_hat: Tensor, Y: Tensor,
                  reduction: Literal['mean', 'none'] = 'mean',
                  eps=1e-12
                  ) -> Tensor | float:
        """Classification loss
        """
        target = Y.argmax(dim=-1)
        loss = -torch.log(Y_hat + eps)[:,:,target]  # NLL
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        raise ValueError


    def criterion(self,
                  Y_hat: Tensor,
                  batch: MaestroDataset.Batch
                  ) -> float:
        mse_loss = AlignerLitModel.metric_RMSE(Y_hat, batch.Y.float())
        cls_loss = AlignerLitModel.metric_CE(Y_hat, batch.Y.float())
        return {
            'mse_loss': mse_loss,
            'cls_loss': cls_loss,
            'loss': mse_loss,
            'nm': AlignerLitModel.metric_NM(Y_hat)
        }


    def forward(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs)


    def training_step(self, batch: MaestroDataset.Batch, **_):
        Y_hat = self.forward(**batch._asdict())
        metrics = self.criterion(Y_hat, batch)
        loss = metrics['mse_loss'] + self.config.alpha_cls * metrics['cls_loss']

        # Monotonicity constraint
        if self.config.nm_penalty is not None and metrics['nm'] != 0:
            loss *= self.config.nm_penalty * metrics['nm']

        self.log_dict({
                'train_loss': loss,
                'train_mse_loss': metrics['mse_loss'],
                'train_cls_loss': metrics['cls_loss'],
                'non_monotonicity': metrics['nm'],
                'sample_idx': self.train_dataset.idx
            }, 
            on_step=True, on_epoch=False,
            sync_dist=True, batch_size=self.config.batch_size
        )

        torch.cuda.empty_cache()
        gc.collect()
        return loss


    @torch.no_grad()
    def validation_step(self, batch: MaestroDataset.Batch, **_):
        Y_hat = self.forward(**batch._asdict())
        metrics = self.criterion(Y_hat, batch)
        loss = metrics['mse_loss'] + self.config.alpha_cls * metrics['cls_loss']
        self.log_dict({
                'val_loss': loss,
                'val_mse_loss': metrics['mse_loss'],
                'val_cls_loss': metrics['cls_loss'],
                'val_non_monotonicity': metrics['nm'],
                'sample_idx': self.val_dataset.idx
            },
            on_step=True, on_epoch=True,
            sync_dist=True, batch_size=self.config.batch_size
        )
        return loss


    def configure_optimizers(self):
        return AdamW(self.model.parameters(), self.config.learning_rate)


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_dataloader_workers,
                          collate_fn=MaestroDataset.collate_fn)


    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_dataloader_workers,
                          collate_fn=MaestroDataset.collate_fn)
