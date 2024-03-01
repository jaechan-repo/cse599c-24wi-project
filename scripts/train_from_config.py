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
import json

import os

torch.cuda.empty_cache()
gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    with open(parser.parse_args().config, 'r') as f:
        config = json.load(f)

    model_config = ModelConfig(**config['model_config'])
    trainer_config = config['trainer_config']

    root_dir = (config['root_dir']
                if config['root_dir'] is not None
                else "")

    if trainer_config['seed'] is not None:
        seed_everything(trainer_config['seed'])

    lit_model_config = LitModelConfig(
        data_dir=os.path.join(root_dir, trainer_config['data_dir']),
        learning_rate=trainer_config['learning_rate'],
        batch_size=trainer_config['batch_size'],
        num_dataloader_workers=0,
        normalization=trainer_config['normalization'],
        loss=trainer_config['loss'],
        shuffle_train=trainer_config['overfit'] is None
    )

    model = AlignerLitModel(model_config, lit_model_config)
    wandb_logger = WandbLogger(project='score-align')
    wandb_logger.watch(model, log="all")

    checkpoint_callback = ModelCheckpoint(
        dirpath=trainer_config['save_to_dir'],
        filename='{epoch}-{step}-{val_loss:.2f}',
        **({'every_n_train_steps': trainer_config['save_every']}
           if trainer_config['save_every_unit'] == 'step'
           else {'every_n_epochs': trainer_config['save_every']}
        )
    )

    overfit = (
        trainer_config['overfit']
        if 0 <= trainer_config['overfit'] < 1
        else int(trainer_config['overfit'])
    )

    trainer = Trainer(
        accelerator=trainer_config['accelerator'],
        max_epochs=trainer_config['max_epochs'],    # infinite
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        default_root_dir=trainer_config['save_to_dir'],
        callbacks=[checkpoint_callback],
        gradient_clip_val=trainer_config['gradient_clip_val'],
        overfit_batches=overfit,
        limit_val_batches= 0 if overfit != 0 else None,
        detect_anomaly=trainer_config['detect_anomaly']
    )

    load_from_dir = (
        find_file(
            os.path.join(
                config['root_dir'],
                trainer_config['load_from_dir']
            ),
            'ckpt'
        )
        if trainer_config['load_from_dir'] is not None
        else None
    )
    trainer.fit(model, ckpt_path=load_from_dir)
