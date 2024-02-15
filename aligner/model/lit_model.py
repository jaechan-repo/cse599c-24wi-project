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


class AlignerLitModel(pl.LightningModule):

    def __init__(self, 
                 model_config: ModelConfig,
                 lit_model_config: LitModelConfig):
        super().__init__()
        self.model = AlignerModel.from_config(model_config)
        self.config = lit_model_config

        # Save hyperparameters
        self.save_hyperparameters(
            lit_model_config._asdict() | model_config._asdict())


    @staticmethod
    def _metric_NM(Y_hat: torch.Tensor) -> float:
        """Ranges from 0 to 1.
        """
        pred_indices = torch.argmax(Y_hat, dim=-1)
        diffs = pred_indices[:, 1:] - pred_indices[:, :-1]
        return torch.mean(torch.any(diffs < 0, dim=-1).float(), dim=-1)


    @staticmethod
    def _metric_EMD(Y_hat: Tensor, Y: Tensor, 
                    reduction: Literal['mse', 'rmse', 'none'] = 'rmse'
                    ) -> Tensor | float:
        """Asymmetric earth mover distance
        """
        assert Y_hat.shape == Y.shape

        target = Y.argmax(dim=-1)
        indices = torch.arange(Y.shape[-1]).view(1,1,-1).type_as(target)
        sq = (indices - target.unsqueeze(-1)) ** 2
        assert sq.shape == Y_hat.shape

        loss = Y_hat * sq.float()   # unreduced
        if reduction == 'none':
            return loss

        loss = loss.sum(dim=-1)
        if reduction == 'mse':
            return loss.mean()
        if reduction == 'rmse':
            return loss.sqrt().mean()


    def criterion(self, Y_hat: Tensor, Y: Tensor) -> float:
        return AlignerLitModel._metric_EMD(Y_hat, Y)


    def forward(self, *args, **kwargs) -> Tensor:
        """See: `aligner.model.AlignerModel`
        """
        return self.model(*args, **kwargs)


    def training_step(self,
                      batch: MaestroDataset.Batch,
                      batch_idx: int):
        Y_hat = self.forward(**batch._asdict())
        loss = self.criterion(Y_hat, batch.Y.int())

        # Monotonicity constraint
        nm: float = AlignerLitModel._metric_NM(Y_hat)
        if self.config.nm_penalty is not None and nm != 0:
            loss *= self.config.nm_penalty * nm

        self.log_dict({
                'train_loss': loss,
                'non_monotonicity': nm
            }, 
            on_step=True, on_epoch=False,
            sync_dist=True, batch_size=self.config.batch_size
        )
        gc.collect()
        return loss


    @torch.no_grad()
    def validation_step(self, batch: MaestroDataset.Batch, **_):
        Y_hat = self.forward(**batch._asdict())
        loss = self.criterion(Y_hat, batch.Y.int())
        self.log(
            "val_loss", loss,
            on_step=True, on_epoch=True,
            sync_dist=True, batch_size=self.config.batch_size
        )
        return loss


    def configure_optimizers(self):
        return AdamW(self.model.parameters(),
                     self.config.learning_rate)


    def train_dataloader(self):
        train_data = MaestroDataset(self.config.data_dir,
                                    split='train')
        return DataLoader(train_data,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_dataloader_workers,
                          collate_fn=MaestroDataset.collate_fn)


    def val_dataloader(self):
        validation_data = MaestroDataset(self.config.data_dir,
                                         split='validation')
        return DataLoader(validation_data,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_dataloader_workers,
                          collate_fn=MaestroDataset.collate_fn)
