import sys
sys.path.append("..")

from aligner.model import AlignerLitModel, ModelConfig, LitModelConfig
from aligner.utils.constants import *
from aligner.utils.utils import find_file

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

import argparse
import gc

torch.cuda.empty_cache()
gc.collect()

if __name__ == "__main__":
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
    parser.add_argument('--normalization', type=str,
                        choices=['none', 'dtw'],
                        default='none')
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--start_at', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overfit', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=None)
    args = parser.parse_args()

    if args.save_every_step is not None \
            and args.save_every_epoch is not None:
        ValueError("Specify only one of --save_every_step or --save_every_epoch.")
    if args.save_every_step is None \
            and args.save_every_epoch is None:
        args.save_every_epoch = 1

    seed_everything(args.seed)

    model_config = ModelConfig(
        ### SCORE ###
        vocab_size=max(TOKEN_ID.values()) + 1,
        d_score=64,
        n_heads_score=4,
        attn_dropout_score=0.1,
        ffn_dropout_score=0.1,
        n_layers_score=4,

        ### AUDIO ###
        d_audio=N_MELS,
        n_heads_audio=8,
        attn_dropout_audio=0.1,
        ffn_dropout_audio=0.1,
        n_layers_audio=4,
    )

    lit_model_config = LitModelConfig(
        data_dir="../data/maestro-v3.0.0",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate if args.learning_rate is not None else 5e-6/16 * args.batch_size,
        num_dataloader_workers=args.num_workers,
        start_at=args.start_at,
        normalization=args.normalization,
        loss=args.loss,
        shuffle_train=args.overfit is None
    )

    model = AlignerLitModel(model_config, lit_model_config)

    wandb_logger = WandbLogger(project='score-align')
    wandb_logger.watch(model, log="all")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_to_path,
        filename='{epoch}-{step}-{val_loss:.2f}',
        every_n_train_steps=args.save_every_step,
        every_n_epochs=args.save_every_epoch
    )

    args.overfit = args.overfit if 0 <= args.overfit < 1 else int(args.overfit)

    trainer = Trainer(
        accelerator=args.accelerator,
        max_epochs=-1,    # infinite
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        log_every_n_steps=5,
        default_root_dir=args.save_to_path,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        overfit_batches= args.overfit if 0 <= args.overfit < 1 else int(args.overfit),
        limit_val_batches= 0 if args.overfit != 0 else None,
        # detect_anomaly=True
    )

    trainer.fit(model, ckpt_path=find_file(
        args.load_from_path, 'ckpt'
    ) if args.load_from_path is not None else None)
