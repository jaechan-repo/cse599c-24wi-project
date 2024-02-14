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

from typing import NamedTuple, Optional, Literal


class TrainerConfig(NamedTuple):
    root_dir: str
    ckpt_dir: str
    learning_rate: float
    training_steps: int
    checking_steps: int
    batch_size: int
    accelerator: Literal['gpu', 'cpu', 'auto'] = 'gpu'
    invalid_pred_penalty: Optional[float] = None
    num_dataloader_workers: int = 1


class AlignerLitModel(pl.LightningModule):

    def __init__(self, 
                 trainer_config: TrainerConfig,
                 model_config: ModelConfig):
        super().__init__()
        self.model = AlignerModel(**model_config._asdict())
        self.criterion_unreduced = nn.BCELoss(reduction='none')
        self.config = trainer_config
        self.save_hyperparameters(trainer_config._asdict() | model_config._asdict(),
                                  ignore=["trainer_config", "model_config"])


    @staticmethod
    def _monotonicity(Y_hat: torch.Tensor) -> bool:
        pred_indices = torch.argmax(Y_hat, dim=-1)
        return torch.all(pred_indices[1:] >= pred_indices[:-1])


    def criterion(self, Y_hat: Tensor, Y: Tensor):
        assert Y_hat.shape == Y.shape, \
                "Input and target must be of the same size."
        loss = self.criterion_unreduced(Y_hat, Y)
        loss = torch.mean(loss)

        # TODO: Prone to bugs when adding negative samples
        if self.config.invalid_pred_penalty is not None \
                and not AlignerLitModel._monotonicity(Y_hat):
            loss *= self.config.invalid_pred_penalty

        return loss


    def forward(self, *args, **kwargs) -> Tensor:
        """See: `aligner.model.AlignerModel`
        """
        return self.model(*args, **kwargs)


    def training_step(self,
                      batch: MaestroDataset.Batch,
                      batch_idx: int):
        Y_hat = self.forward(**batch._asdict())

        # Save to ckpt path
        if (batch_idx + 1) % (self.config.checking_steps / self.config.batch_size) == 0:
            torch.save(self.model.state_dict(),
                       self.config.ckpt_dir + '/' + str((batch_idx+1) * self.config.batch_size) + '.ckpt')

        loss = self.criterion(Y_hat, batch.Y.float())
        self.log("train/loss",
                 loss,
                 sync_dist=True,
                 batch_size=self.config.batch_size)
        return loss


    @torch.no_grad()
    def validation_step(self, batch: MaestroDataset.Batch, **_):
        Y_hat = self.forward(**batch._asdict())
        loss = self.criterion(Y_hat, batch.Y.float())
        self.log("train/loss", loss,
                 sync_dist=True,
                 batch_size=self.config.batch_size)
        return loss


    def configure_optimizers(self):
        return AdamW(self.model.parameters(),
                     self.config.learning_rate)


    def train_dataloader(self):
        train_data = MaestroDataset(self.config.root_dir,
                                    split='train')
        return DataLoader(train_data,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_dataloader_workers,
                          collate_fn=MaestroDataset.collate_fn)


    def val_dataloader(self):
        validation_data = MaestroDataset(self.config.root_dir, split='validation')
        return DataLoader(validation_data,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_dataloader_workers,
                          collate_fn=MaestroDataset.collate_fn)
