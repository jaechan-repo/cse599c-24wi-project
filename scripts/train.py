import sys
sys.path.append("..")

from aligner.model import AlignerLitModel
from aligner.utils import set_seed
from aligner.utils.constants import *
from aligner.utils.config import model_config, trainer_config

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import gc


if __name__ == "__main__":
    set_seed(42)
    torch.cuda.empty_cache()
    gc.collect()

    model = AlignerLitModel(trainer_config, model_config)
    print(model)

    wandb_logger = WandbLogger(project='score-align')
    trainer = Trainer(accelerator=trainer_config.accelerator,
                      devices='auto',
                      logger=wandb_logger,
                      check_val_every_n_epoch=trainer_config.checking_steps,
                      max_steps=trainer_config.training_steps)
    trainer.fit(model)
