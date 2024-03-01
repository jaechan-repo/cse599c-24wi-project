import torch
from torch.optim import AdamW
from ..utils.scheduler import CosineAnnealingWarmupRestarts, LinearWarmupCosineAnnealingLR
from torch.utils.data import DataLoader
from torch import Tensor
import pytorch_lightning as pl

from ..dataset import MaestroDataset, Batch
from ..inference.metrics import monotonicity, compute_metrics
from aligner.utils.constants import *
from aligner.model import AlignerModel, ModelConfig
from aligner.utils.constants import *

from typing import NamedTuple, Literal
import gc


class LitModelConfig(NamedTuple):
    data_dir: str
    learning_rate: float
    batch_size: int
    num_dataloader_workers: int = 0
    alpha_cls: float = 1.0
    start_at: int = 0
    normalization: Literal['none', 'dtw'] = 'none'
    loss: str = 'cross_entropy'
    shuffle_train: bool = True


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
                                            shuffle=True)
        self.val_dataset = MaestroDataset(self.config.data_dir,
                                          split='validation')


    def criterion(self,
                  Y_hat: Tensor,
                  batch: Batch
                  ) -> float:
        Y = batch.Y.float()
        n_events = batch.event_padding_mask.float().sum(dim=-1)
        metrics = compute_metrics(Y_hat, Y, n_events)
        return metrics


    def forward(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs,
                          normalization=self.config.normalization)


    def training_step(self, batch: Batch, **_):
        Y_hat = self.forward(**batch._asdict())
        metrics = self.criterion(Y_hat, batch)
        self.log_dict({
                'train_loss': metrics[self.config.loss],
                'train_cross_entropy': metrics['cross_entropy'],
                # 'train_rmse_loss': metrics['rmse_loss'],
                # 'train_rmse_loss_normalized': metrics['rmse_loss_normalized'],
                'emd_loss': metrics['emd_loss'],
                'structured_perceptron_loss': metrics['structured_perceptron_loss'],
                'train_monotonicity': metrics['monotonicity'],
                'train_sample_idx': self.train_dataset.idx
            },
            on_step=True, on_epoch=False,
            sync_dist=True, batch_size=self.config.batch_size
        )
        gc.collect()
        return metrics[self.config.loss]


    @torch.no_grad()
    def validation_step(self, batch: Batch, **_):
        Y_hat = self.forward(**batch._asdict())
        metrics = self.criterion(Y_hat, batch)
        self.log_dict({
                'val_loss': metrics[self.config.loss],
                'val_monotonicity': metrics['monotonicity'],
                'val_sample_idx': self.val_dataset.idx,
            },
            on_step=True, on_epoch=True,
            sync_dist=True, batch_size=self.config.batch_size
        )
        return metrics[self.config.loss]


    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.config.learning_rate)
        # lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
        #                                              first_cycle_steps=400,
        #                                              max_lr=self.config.learning_rate,
        #                                              min_lr=self.config.learning_rate/1e3,
        #                                              warmup_steps=200,
        #                                              )
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                     warmup_epochs=2000,
                                                     max_epochs=200000,
                                                     warmup_start_lr=self.config.learning_rate,
                                                     eta_min=self.config.learning_rate / 1e3)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }}


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
