import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor, LongTensor
from typing import Optional, NamedTuple, Literal, List
from ..utils.constants import *
from .attention import MultiheadAttention, MultiheadSelfAttention
from .dtw import BidirectionalHardDTW
from .pos_encoding import RotaryEmbedding


class ModelConfig(NamedTuple):
    vocab_size: int
    d_score: int
    n_heads_score: int
    d_audio: int
    attn_dropout_score: float
    ffn_dropout_score: float
    n_layers_score: int
    n_heads_audio: int
    attn_dropout_audio: float
    ffn_dropout_audio: float
    n_layers_audio: int


class ScoreEncoderBlock(nn.Module):

    def __init__(self,
                 d_embed: int,
                 n_heads: int,
                 attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.5,
                 ffn_expansion: int = 4):
        super().__init__()

        self.LocalAttn = MultiheadSelfAttention(d_embed, n_heads, dropout=attn_dropout)
        self.GlobalAttn = MultiheadSelfAttention(d_embed, n_heads, dropout=attn_dropout)

        self.Norm1 = nn.LayerNorm(d_embed)
        self.Norm2 = nn.LayerNorm(d_embed)
        self.Norm3 = nn.LayerNorm(d_embed)

        self.FFN = nn.Sequential(
            nn.Linear(d_embed, ffn_expansion * d_embed),
            nn.GELU(),
            nn.Linear(ffn_expansion * d_embed, d_embed),
            nn.Dropout(ffn_dropout)
        )


    def forward(self,
                x: Tensor,
                local_attn_mask: BoolTensor | None = None,
                global_attn_mask: BoolTensor | None = None
                ) -> Tensor:
        x = x + self.LocalAttn(self.Norm1(x), local_attn_mask)
        x = x + self.GlobalAttn(self.Norm2(x), global_attn_mask)
        x = x + self.FFN(self.Norm3(x))
        assert not x.isnan().any()
        return x


class ScoreEncoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_score: int,
                 n_heads: int,
                 attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.5,
                 ffn_expansion: int = 4,
                 n_layers: int = 5):
        super().__init__()
        self.Lookup = nn.Embedding(vocab_size, d_score,
                                   padding_idx=TOKEN_ID['[PAD]'])
        self.EncoderBlocks = nn.ModuleList([
            ScoreEncoderBlock(
                d_score, n_heads, attn_dropout, ffn_dropout, ffn_expansion
            ) for _ in range(n_layers)
        ])


    def forward(self,
                input_ids: LongTensor,
                local_attn_mask: BoolTensor | None = None,
                global_attn_mask: BoolTensor | None = None
                ) -> Tensor:
        x: Tensor = self.Lookup(input_ids)
        for Block in self.EncoderBlocks:
            x = Block(x, local_attn_mask, global_attn_mask)

        assert not x.isnan().any()
        return x
  

class AudioEncoderBlock(nn.Module):

    def __init__(self,
                 d_audio: int,
                 d_score: int,
                 n_heads: int,
                 attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.5,
                 ffn_expansion: int = 4):
        super().__init__()

        self.SelfAttn = MultiheadSelfAttention(d_audio, n_heads, dropout=attn_dropout)
        self.XAttn = MultiheadAttention(d_audio, d_score, n_heads, dropout=attn_dropout)

        self.Norm1 = nn.LayerNorm(d_audio)
        self.Norm2 = nn.LayerNorm(d_audio)
        self.Norm3 = nn.LayerNorm(d_audio)
        self.FFN = nn.Sequential(
            nn.Linear(d_audio, ffn_expansion * d_audio),
            nn.GELU(),
            nn.Linear(ffn_expansion * d_audio, d_audio),
            nn.Dropout(ffn_dropout)
        )


    def forward(self, 
                x_audio: Tensor, 
                x_event: Tensor,
                event_padding_mask: Optional[BoolTensor] = None):

        x_audio = x_audio + self.SelfAttn(self.Norm1(x_audio))
        x_audio = x_audio + self.XAttn(self.Norm2(x_audio),
                                       x_event,
                                       event_padding_mask.unsqueeze(1))
        x_audio = self.FFN(self.Norm3(x_audio))

        assert not x_audio.isnan().any()
        return x_audio
    

class AudioEncoder(nn.Module):

    def __init__(self,
                 d_audio: int,
                 d_score: int,
                 n_heads: int,
                 attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.5,
                 ffn_expansion: int = 4,
                 n_layers: int = 5):
        super().__init__()

        self.GELU = nn.GELU()
        self.Embed = nn.Linear(d_audio, d_audio)
        self.EncoderBlocks = nn.ModuleList([AudioEncoderBlock(
            d_audio, d_score, n_heads,
            attn_dropout, ffn_dropout, ffn_expansion
        ) for _ in range(n_layers)])


    def forward(self,
                audio_frames: Tensor,
                event_embed: Tensor,
                event_padding_mask: Optional[BoolTensor] = None
                ) -> Tensor:
        x = self.GELU(self.Embed(audio_frames))
        for Block in self.EncoderBlocks:
            x = Block(x, event_embed, event_padding_mask)
        assert not x.isnan().any()
        return x


class CrossAttentionHead(nn.Module):

    def __init__(self, d_audio: int, d_score: int, bias=True):
        super().__init__()
        self.d_h = min(d_audio, d_score)   # min(128, 64) // 2 = 32
        self.RotaryEmbedding = RotaryEmbedding(dim=self.d_h // 2)
        self.Rotate = lambda x: self.RotaryEmbedding.rotate_queries_or_keys(x)
        self.Linear_Q = nn.Linear(d_audio, self.d_h, bias=bias)
        self.Linear_K = nn.Linear(d_score, self.d_h, bias=bias)


    def forward(self,
                x_audio: Tensor,
                x_event: Tensor,
                event_padding_mask: BoolTensor | None = None
                ) -> Tensor:
        Q = self.Linear_Q(x_audio)
        K = self.Linear_K(x_event)
        Q, K = self.Rotate(Q), self.Rotate(K)
        attn = Q @ K.transpose(1, 2)
        # assert attn.shape == (batch_size, n_frames, max_n_events)

        if event_padding_mask is not None:
            event_padding_mask = event_padding_mask.unsqueeze(1)
            attn = attn.masked_fill((~event_padding_mask).to(attn.device),
                                    float('-inf'))

        assert not attn.isnan().any()
        return attn


class AlignerModel(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_score: int, n_heads_score: int,
                 d_audio: int, n_heads_audio: int, 
                 attn_dropout_score: float = 0.0,
                 ffn_dropout_score: float = 0.5,
                 n_layers_score: int = 5,
                 attn_dropout_audio: float = 0.0,
                 ffn_dropout_audio: float = 0.5,
                 n_layers_audio: int = 5,
                 **_):
        """
        Args:
            vocab_size (int): Size of the vocabulary for the score encoder.
            d_score (int): Embedding dimension for the score encoder.
            n_heads_score (int): Number of attention heads for the score encoder.
            d_audio (int): Embedding dimension for the audio encoder.
            n_heads_audio (int): Number of attention heads for the audio encoder.
            attn_dropout_score (float, optional): Attention dropout probability
                    for the score encoder. Defaults to 0.0.
            ffn_dropout_score (float, optional): Feedforward network dropout
                    probability for the score encoder. Defaults to 0.5.
            n_layers_score (int, optional): Number of transformer blocks to stack
                    for the score encoder. Defaults to 5.
            attn_dropout_audio (float, optional): Attention dropout probability
                    for the audio encoder. Defaults to 0.0.
            ffn_dropout_audio (float, optional): Feedforward network dropout
                    probability for the audio encoder. Defaults to 0.5.
            n_layers_audio (int, optional): Number of transformer blocks to stack
                    for the audio encoder. Defaults to 5.
        """
        super(AlignerModel, self).__init__()
        self.ScoreEncoder = ScoreEncoder(vocab_size, d_score, n_heads_score,
                                         attn_dropout_score, ffn_dropout_score,
                                         n_layers_score)
        self.AudioEncoder = AudioEncoder(d_audio, d_score, n_heads_audio,
                                         attn_dropout_audio, ffn_dropout_audio,
                                         n_layers_audio)
        self.Head = CrossAttentionHead(d_audio, d_score)
        self.DTW = BidirectionalHardDTW()


    @classmethod
    def from_config(cls, config: ModelConfig):
        return cls(**config._asdict())


    def forward(self,
                audio_frames: Tensor,
                score_ids: LongTensor,
                local_event_attn_mask: BoolTensor,
                global_event_attn_mask: BoolTensor,
                score_to_event: BoolTensor,
                event_padding_mask: BoolTensor,
                normalization: Literal['none', 'dtw'],
                **_) -> Tensor:
        score_embed: Tensor = self.ScoreEncoder(score_ids,
                                                local_event_attn_mask,
                                                global_event_attn_mask)
        event_embed = score_to_event.float() @ score_embed
        audio_embed: Tensor = self.AudioEncoder(audio_frames,
                                                event_embed,
                                                event_padding_mask)
        out = self.Head(audio_embed, event_embed, event_padding_mask)

        if normalization == 'dtw':
            out = self.DTW(out)
        else:
            assert normalization == 'none'

        assert not out.isnan().any()
        return out


    def forward_unbatched(self,
                          audio_frames: Tensor,
                          score_ids: LongTensor,
                          local_event_attn_mask: BoolTensor,
                          global_event_attn_mask: BoolTensor,
                          score_to_event: BoolTensor,
                          normalization: Literal['dtw', 'none'],
                          device='cuda',
                          **_) -> Tensor:
        assert len(score_ids.shape) == 1, "Your input is batched."

        out = self(audio_frames.unsqueeze(0).to(device),
                   score_ids.unsqueeze(0).to(device),
                   local_event_attn_mask.unsqueeze(0).to(device),
                   global_event_attn_mask.unsqueeze(0).to(device),
                   score_to_event.unsqueeze(0).to(device),
                   event_padding_mask=torch.ones((1, score_to_event.shape[0]),
                                                 dtype=torch.bool),
                   normalization=normalization).to(device)

        return out.squeeze(0)
