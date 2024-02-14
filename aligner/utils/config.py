from ..model import ModelConfig, TrainerConfig
from .constants import *


trainer_config = TrainerConfig(
    root_dir="../data/maestro-v3.0.0",
    ckpt_dir="../ckpt",
    learning_rate=1e-4,
    training_steps=10**6,
    checking_steps=10**2,
    batch_size=4,
    accelerator='gpu',
    invalid_pred_penalty=1000,
    num_dataloader_workers=4
)


model_config = ModelConfig(
    vocab_size=max(TOKEN_ID.values()) + 1,
    d_score=64,
    n_heads_score=4,
    d_audio=N_MELS,
    attn_dropout_score=0.0,
    ffn_dropout_score=0.5,
    n_layers_score=5,
    n_heads_audio=8,
    attn_dropout_audio=0.0,
    ffn_dropout_audio=0.5,
    n_layers_audio=5
)
