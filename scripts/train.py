import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch import Tensor, LongTensor
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from aligner.dataset import MaestroDataset
from aligner.utils.constants import *
from aligner.model import ScoreAlign
from aligner.utils.constants import *
from aligner.utils.metrics import monotonicity
# from aligner.config.data_config import data_config
# from aligner.config.model_config import model_config

from typing import NamedTuple
import gc


device = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = 1


class TrainerConfig(NamedTuple):
    learning_rate: float
    training_steps: int
    checking_steps: int
    batch_size: int

trainer_config = TrainerConfig(
    learning_rate=1e-4,
    training_steps=10**6,
    checking_steps=10*5,    # # steps after which a training checkpoint is saved
    batch_size=1
)


class AlignTrainer(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.model = ScoreAlign()
        self.criterion_unreduced = nn.BCELoss(reduction='none')


    def criterion(self, Y_hat_b: Tensor, Y_b: Tensor, midi_event_timestamps: Tensor):
        assert Y_hat_b.shape == Y_b.shape
        assert Y_hat_b.dtype == Y_b.dtype

        loss = self.criterion_unreduced(Y_hat_b, Y_b)
        assert loss.shape[0] == trainer_config.batch_size
        assert loss.shape[1] == N_FRAMES_PER_CLIP

        # Monotonicity constraint
        assert monotonicity(Y_hat_b, midi_event_timestamps)

        loss = torch.mean(torch.sum(torch.mean(loss, dim=-1), dim=-1), dim=-1)
        return loss


    # TODO: Fix according to the API suggested by the model
    def forward(self,
                audio_frames: Tensor,
                score_ids: LongTensor,
                score_attn_mask: LongTensor,
                xattn_mask: LongTensor
                ) -> Tensor:
        """
        Args:
            audio_frames (Tensor): Audio input representation.
                Size: (B, N_FRAMES_PER_CLIP, N_MELS)
            score_ids (LongTensor): Score input represnetation.
                Size: (B, max_n_tokens)
            score_attn_mask (LongTensor): Attention mask to be applied to the
                self-attention layers of the score encoder.
                Size: (B, max_n_tokens, max_n_tokens).
            xattn_mask (LongTensor): Attention mask to be applied to the
                intermediate cross-attention layers and the output layer.
                Size: (B, N_FRAMES_PER_CLIP, max_n_tokens).

        Returns:
            Tensor: Alignment matrix Y_hat.
                    Size: (B, N_FRAMES_PER_CLIP, max_n_tokens).
        """
        return self.model(audio_frames, score_ids, score_attn_mask, xattn_mask)


    def training_step(self,
                      batch: MaestroDataset.Batch,
                      batch_idx: int):

        xattn_mask: LongTensor = batch.score_event_mask_b.repeat(0, N_FRAMES_PER_CLIP)
        assert xattn_mask.shape == batch.Y_b.shape

        # TODO: Fix according to the API suggested by the model
        Y_hat_b = self.forward(
            audio_frames=batch.audio_frames_b,
            score_ids=batch.score_ids_b,
            score_attn_mask=batch.score_attn_mask_b,
            xattn_mask=xattn_mask,
        )

        if (batch_idx + 1) % (trainer_config.checking_steps / trainer_config.batch_size) == 0:
            torch.save(model.state_dict(), self.cpt_path + '/' + str((batch_idx+1) * trainer_config.batch_size) + '.ckpt')

        loss = self.criterion(Y_hat_b, batch.Y_b.float(), batch.midi_event_timestamps)
        self.log("train/loss", loss)
        return loss


    @torch.no_grad()
    def validation_step(self, 
                        batch: MaestroDataset.Batch,
                        batch_idx: int):
        
        xattn_mask: LongTensor = batch.score_event_mask_b.repeat(0, N_FRAMES_PER_CLIP)
        assert xattn_mask.shape == batch.Y_b.shape

        # TODO: Fix according to the API suggested by the model
        Y_hat_b = self.forward(
            audio_frames=batch.audio_frames_b,
            score_ids=batch.score_ids_b,
            score_attn_mask=batch.score_attn_mask_b,
            xattn_mask=xattn_mask
        )

        loss = self.criterion(Y_hat_b, batch.Y_b.float(), batch.midi_event_timestamps)
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
