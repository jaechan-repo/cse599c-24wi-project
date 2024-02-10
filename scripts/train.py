import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch import Tensor, LongTensor, BoolTensor
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from aligner.dataset import MaestroDataset
from aligner.utils.constants import *
from aligner.model import AlignerModel, ModelConfig
from aligner.utils.constants import *

from typing import NamedTuple
import gc


device = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = 1


class TrainerConfig(NamedTuple):
    learning_rate: float = 1e-4
    training_steps: int = 10**6
    checking_steps: int = 10**5
    batch_size: int = 1

trainer_config = TrainerConfig()


class AlignTrainer(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model_config = ModelConfig()
        self.model = AlignerModel(**self.model_config._asdict())
        self.criterion_unreduced = nn.BCELoss(reduction='none')


    def criterion(self, Y_hat: Tensor, Y: Tensor):
        assert Y_hat.shape == Y.shape
        assert Y_hat.dtype == Y.dtype

        loss = self.criterion_unreduced(Y_hat, Y)
        assert loss.shape[0] == trainer_config.batch_size
        assert loss.shape[1] == N_FRAMES_PER_CLIP

        # loss = torch.mean(torch.sum(torch.mean(loss, dim=-1), dim=-1), dim=-1)
        loss = torch.mean(loss)
        return loss


    def forward(self,
                audio_frames: Tensor,
                score_ids: LongTensor,
                score_attn_mask: BoolTensor,
                event_padding_mask: BoolTensor
                ) -> Tensor:
        """See: `aligner.model.AlignerModel`
        """
        return self.model(audio_frames, score_ids,
                          score_attn_mask, event_padding_mask)


    def training_step(self,
                      batch: MaestroDataset.Batch,
                      batch_idx: int):

        Y_hat_b = self.forward(
            audio_frames=batch.audio_frames_b,
            score_ids=batch.score_ids_b,
            score_attn_mask=batch.score_attn_mask_b,
            event_padding_mask=batch.event_padding_mask_b
        )

        if (batch_idx + 1) % (trainer_config.checking_steps / trainer_config.batch_size) == 0:
            torch.save(model.state_dict(), self.cpt_path + '/' + str((batch_idx+1) * trainer_config.batch_size) + '.ckpt')

        loss = self.criterion(Y_hat_b, batch.Y_b.float())
        self.log("train/loss", loss)
        return loss


    @torch.no_grad()
    def validation_step(self, batch: MaestroDataset.Batch, *_):

        Y_hat_b = self.forward(
            audio_frames=batch.audio_frames_b,
            score_ids=batch.score_ids_b,
            score_attn_mask=batch.score_attn_mask_b,
            event_padding_mask=batch.event_padding_mask_b
        )

        loss = self.criterion(Y_hat_b, batch.Y_b.float())
        self.log("train/loss", loss)
        return loss


    def configure_optimizers(self):
        return AdamW(self.model.parameters(),
                     trainer_config.learning_rate)


    def train_dataloader(self, root_dir: str):
        train_data = MaestroDataset(root_dir, split='train')
        return DataLoader(train_data,
                          batch_size=trainer_config.batch_size,
                          num_workers=4,
                          collate_fn=MaestroDataset.collate_fn)


    def val_dataloader(self, root_dir: str):
        validation_data = MaestroDataset(root_dir, split='validation')
        return DataLoader(validation_data,
                          batch_size=trainer_config.batch_size,
                          num_workers=4,
                          collate_fn=MaestroDataset.collate_fn)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()

    model = AlignTrainer()

    print(model)
    print(trainer_config)

    wandb_logger = WandbLogger(project="score-align")
    trainer = pl.Trainer(gpus=num_gpus,
                         logger=wandb_logger,
                         check_val_every_n_epoch=trainer_config.checking_steps,
                         max_steps=trainer_config.training_steps
                         )
    trainer.fit(model)
