import sys
sys.path.append("..")

from aligner.model import AlignerLitModel, ModelConfig, LitModelConfig
from aligner.utils import set_seed
from aligner.utils.constants import *

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

import argparse
import gc

torch.cuda.empty_cache()
gc.collect()


model_config = ModelConfig(
    ### SCORE ###
    vocab_size=max(TOKEN_ID.values()) + 1,
    d_score=64,
    n_heads_score=4,
    attn_dropout_score=0.0,
    ffn_dropout_score=0.5,
    n_layers_score=4,

    ### AUDIO ###
    d_audio=N_MELS,
    n_heads_audio=8,
    attn_dropout_audio=0.1,
    ffn_dropout_audio=0.5,
    n_layers_audio=4,
)


if __name__ == "__main__":
    """
    Example run:
    python train.py --save_to_path ../ckpt/v0/01 --save_every_epoch 1 -- batch_size 8
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_to_path', type=str, default=None,
        help="Trainer saves checkpoint to this path. Defaults to current cwd."
    )
    parser.add_argument('--load_from_path', type=str, default=None,
        help="Trainer loads checkpoint from this path. Defaults to not loading."
    )
    parser.add_argument('--save_every_step', type=int, default=None,
        help="Trainer saves every given number of training steps."
    )
    parser.add_argument('--save_every_epoch', type=int, default=None,
        help="Trainer saves every given number of epochs."
    )
    parser.add_argument('--start_at', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--nm_penalty', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.save_every_step is not None \
            and args.save_every_epoch is not None:
        ValueError("Specify only one of --save_every_step or --save_every_epoch.")

    if args.save_every_step is None \
            and args.save_every_epoch is None:
        args.save_every_epoch = 1

    seed_everything(args.seed)

    lit_model_config = LitModelConfig(
        data_dir="../data/maestro-v3.0.0",
        batch_size=args.batch_size,
        learning_rate=1e-4/16 * args.batch_size,
        nm_penalty=args.nm_penalty,
        num_dataloader_workers=args.num_workers,
        start_at=args.start_at
    )

    model = AlignerLitModel(model_config, lit_model_config)

    wandb_logger = WandbLogger(project='score-align')
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_to_path,
        filename='{epoch}-{step}-{val_loss:.2f}',
        every_n_train_steps=args.save_every_step,
        every_n_epochs=args.save_every_epoch
    )

    trainer = Trainer(
        accelerator=args.accelerator,
        max_epochs=-1,    # infinite
        check_val_every_n_epoch=1,
        limit_val_batches=200,
        logger=wandb_logger,
        log_every_n_steps=50,
        default_root_dir=args.save_to_path,
        callbacks=[checkpoint_callback],
        # deterministic=True,   # Slows down
    )

    trainer.fit(model, ckpt_path=args.load_from_path)
