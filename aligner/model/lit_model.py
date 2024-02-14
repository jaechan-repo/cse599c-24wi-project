import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch import Tensor
import pytorch_lightning as pl

from ..dataset import MaestroDataset
from aligner.utils.constants import *
from aligner.model import AlignerModel, ModelConfig
from aligner.utils.constants import *

from typing import NamedTuple, Optional
import gc


class LitModelConfig(NamedTuple):
    data_dir: str
    learning_rate: float
    batch_size: int
    invalid_pred_penalty: Optional[float] = None
    num_dataloader_workers: int = 0


class AlignerLitModel(pl.LightningModule):

    def __init__(self, 
                 model_config: ModelConfig,
                 lit_model_config: LitModelConfig):
        super().__init__()
        self.model = AlignerModel.from_config(model_config)
        self.criterion_unreduced = nn.BCELoss(reduction='none')
        self.config = lit_model_config

        # Save hyperparameters
        self.save_hyperparameters(
            lit_model_config._asdict() | model_config._asdict())


    @staticmethod
    def _monotonicity(Y_hat: torch.Tensor) -> bool:
        """Designed for batch processing
        """
        pred_indices = torch.argmax(Y_hat, dim=-1)
        diffs = pred_indices[:, 1:] - pred_indices[:, :-1]
        return torch.all(diffs >= 0)


    def criterion(self, Y_hat: Tensor, Y: Tensor, padding_mask: Tensor) -> float:
        assert Y_hat.shape == Y.shape, \
                "Input and target must be of the same size."

        n_events_b = padding_mask.sum()     # Total number of events, across the batches
        loss = self.criterion_unreduced(Y_hat, Y)
        loss = torch.sum(loss, dim=-1)      # Sum probabilities for events  (reflected in `n_events_b`)
        loss = torch.mean(loss, dim=-1)     # Mean probabilities for audio
        loss = torch.sum(loss, dim=-1)      # Sum probabilities for batches (reflected in `n_events_b`)
        loss /= n_events_b
        return loss


    def forward(self, *args, **kwargs) -> Tensor:
        """See: `aligner.model.AlignerModel`
        """
        return self.model(*args, **kwargs)


    def training_step(self,
                      batch: MaestroDataset.Batch,
                      batch_idx: int):
        Y_hat = self.forward(**batch._asdict())
        loss = self.criterion(Y_hat, batch.Y.float(), batch.event_padding_mask)

        # Monotonicity constraint.
        # TODO: Revisit for v1.
        monotonic: int
        if self.config.invalid_pred_penalty is None:
            monotonic = -1
        else:
            if AlignerLitModel._monotonicity(Y_hat):
                monotonic = 1
            else:
                # loss *= self.config.invalid_pred_penalty
                monotonic = 0

        self.log_dict({
                'train_loss': loss,
                'monotonic': monotonic
            }, 
            on_step=True, on_epoch=False,
            sync_dist=True, batch_size=self.config.batch_size
        )
        gc.collect()
        return loss


    @torch.no_grad()
    def validation_step(self, batch: MaestroDataset.Batch, **_):
        Y_hat = self.forward(**batch._asdict())
        loss = self.criterion(Y_hat, batch.Y.float(), batch.event_padding_mask)
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
