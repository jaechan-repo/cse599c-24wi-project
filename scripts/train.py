import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import gc

from aligner.data_loader.data_loader import MaestroDataset
from aligner.utils.constants import *
from aligner.config.data_config import data_config
from aligner.config.model_config import model_config
from aligner.model import ScoreAlign


device = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = 1

experiment_config={
        "learning_rate": 1e-4,
        "training_steps": 1000000,
        "checking_steps": 100000, # number of steps after which a training checkpoint is saved
        "batch_size": 1
        }

class AlignTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ScoreAlign(config=model_config)
        self.criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_ID['[PAD]'])
        self.cpt_path = data_config.cpt_path

    def forward(self, encoder_input_tokens, decoder_target_tokens):
        return self.model.forward(encoder_input_tokens, decoder_target_tokens, decoder_input_tokens=None)
    
    def training_step(self, batch, batch_idx):
        inputs = batch['inputs']
        targets = batch['targets']
        outputs = self.forward(encoder_input_tokens=inputs, decoder_target_tokens=targets)
        loss = self.criterion(outputs.permute(0,2,1), targets)
        
        if (batch_idx + 1) % (experiment_config['checking_steps'] / experiment_config['batch_size']) == 0:
            torch.save(model.state_dict(), self.cpt_path + '/' + str((batch_idx+1) * experiment_config['batch_size']) + '.ckpt')
        
        self.log("train/loss", loss)
        
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs = batch['inputs']
        targets = batch['targets']
        outputs = self.forward(encoder_input_tokens=inputs, decoder_target_tokens=targets)
        loss = self.criterion(outputs.permute(0,2,1), targets)

        self.log("train/loss", loss)

        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), experiment_config['learning_rate'])
        
        return optimizer
    
    def train_dataloader(self, root_dir):
        train_data = MaestroDataset(root_dir, split='train')
        trainloader = DataLoader(train_data, batch_size=experiment_config["batch"], num_workers=4)
        return trainloader

    
    def val_dataloader(self, root_dir):
        validation_data = MaestroDataset(root_dir, split='validation')
        validloader = DataLoader(validation_data, batch_size=experiment_config["batch"], num_workers=4)
        return validloader
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    
    model = AlignTrainer()
    
    print(model)
    print(experiment_config)

    wandb_logger = WandbLogger(project="score-align")
    trainer = pl.Trainer(gpus=num_gpus,
                         logger=wandb_logger,
                         check_val_every_n_epoch=experiment_config["checking_steps"],
                         max_steps=experiment_config['training_steps']
                         )
    trainer.fit(model)