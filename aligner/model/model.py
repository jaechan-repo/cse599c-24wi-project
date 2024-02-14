import torch
import torch.nn as nn
from torch import Tensor, BoolTensor, LongTensor
from typing import Optional, NamedTuple
from ..utils.constants import *
from .attention import MultiheadAttention, MultiheadALiBiSelfAttention
import math


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
        super(ScoreEncoderBlock, self).__init__()

        self.SelfAttn = MultiheadALiBiSelfAttention(d_embed, n_heads, attn_dropout)

        self.Norm1 = nn.LayerNorm(d_embed)
        self.Norm2 = nn.LayerNorm(d_embed)

        self.FFN = nn.Sequential(
            nn.Linear(d_embed, ffn_expansion * d_embed),
            nn.ReLU(),
            nn.Linear(ffn_expansion * d_embed, d_embed),
        )

        self.Dropout = nn.Dropout(ffn_dropout)


    def forward(self,
                embed: Tensor,
                attn_mask: Optional[Tensor] = None
                ) -> Tensor:
        out1 = self.SelfAttn(embed, attn_mask)
        out1 = self.Dropout(self.Norm1(out1 + embed))

        out: Tensor = self.FFN(out1)
        out = self.Dropout(self.Norm2(out + out1))

        assert not out.isnan().any()
        return out


class ScoreEncoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_score: int,
                 n_heads: int,
                 attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.5,
                 ffn_expansion: int = 4,
                 n_layers: int = 5):
        super(ScoreEncoder, self).__init__()
        self.Lookup = nn.Embedding(vocab_size, d_score,
                                         padding_idx=TOKEN_ID['[PAD]'])

        self.EncoderBlocks = nn.ModuleList([
            ScoreEncoderBlock(
                d_score, n_heads, attn_dropout, ffn_dropout, ffn_expansion
            ) for _ in range(n_layers)
        ])


    def forward(self,
                input_ids: LongTensor,
                attn_mask: Optional[BoolTensor] = None
                ) -> Tensor:
        out: Tensor = self.Lookup(input_ids)
        for Block in self.EncoderBlocks:
            out = Block(out, attn_mask)

        assert not out.isnan().any()
        return out
  

class AudioEncoderBlock(nn.Module):

    def __init__(self,
                 d_audio: int,
                 d_score: int,
                 n_heads: int,
                 attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.5,
                 ffn_expansion: int = 4):
        super(AudioEncoderBlock, self).__init__()

        self.SelfAttn = MultiheadALiBiSelfAttention(d_audio, n_heads, attn_dropout)
        self.XAttn = MultiheadAttention(d_audio, d_score, n_heads,
                                        dropout=attn_dropout)

        self.Norm1 = nn.LayerNorm(d_audio)
        self.Norm2 = nn.LayerNorm(d_audio)
        self.Norm3 = nn.LayerNorm(d_audio)
        self.FFN = nn.Sequential(
            nn.Linear(d_audio, ffn_expansion * d_audio),
            nn.ReLU(),
            nn.Linear(ffn_expansion * d_audio, d_audio),
        )
        self.Dropout = nn.Dropout(ffn_dropout)


    def forward(self, 
                audio_embed: Tensor, 
                event_embed: Tensor,
                event_padding_mask: Optional[BoolTensor] = None):

        ### Step 1: Self-attention ###
        out1 = self.SelfAttn(audio_embed)
        out1 = self.Dropout(self.Norm1(out1 + audio_embed)) # residual
        
        ### Step 2: Cross-attention ###
        _, n_frames, _ = audio_embed.shape
        attn_mask = event_padding_mask.unsqueeze(1).repeat(1, n_frames, 1)
        out2 = self.XAttn(audio_embed, event_embed, attn_mask)
        out2 = self.Dropout(self.Norm2(out2 + out1))

        ### Step 3: Feedforward network ###
        out = self.FFN(out2)
        out = self.Dropout(self.Norm3(out + out2))

        assert not out.isnan().any()
        return out
    

class AudioEncoder(nn.Module):

    def __init__(self,
                 d_audio: int,
                 d_score: int,
                 n_heads: int,
                 attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.5,
                 ffn_expansion: int = 4,
                 n_layers: int = 5):
        super(AudioEncoder, self).__init__()

        self.EncoderBlocks = nn.ModuleList([AudioEncoderBlock(
            d_audio, d_score, n_heads,
            attn_dropout, ffn_dropout, ffn_expansion
        ) for _ in range(n_layers)])


    def forward(self,
                audio_frames: Tensor,
                event_embed: Tensor,
                event_padding_mask: Optional[BoolTensor] = None
                ) -> Tensor:
        out = audio_frames
        for Block in self.EncoderBlocks:
            out = Block(out, event_embed, event_padding_mask)

        assert not out.isnan().any()
        return out


class CrossAttentionHead(nn.Module):
    """Alignment head implemented as cross-attention.
    The only learnable components are the query and key projectors.
    """

    def __init__(self, d_audio: int, d_score: int, bias=True):
        super().__init__()
        self.Linear_Q = nn.Linear(d_audio, d_audio, bias=bias)  # projects into d_score
        self.Linear_K = nn.Linear(d_score, d_audio, bias=bias)
        self.scale = 1 / math.sqrt(d_audio)


    def forward(self,
                audio_embed: Tensor,
                event_embed: Tensor,
                event_padding_mask: Optional[BoolTensor] = None
                ) -> Tensor:

        _, n_frames, _ = audio_embed.shape

        Q: Tensor = self.Linear_Q(audio_embed)
        K: Tensor = self.Linear_K(event_embed)
        attn = Q @ K.transpose(1, 2) * self.scale
        # assert attn.shape == (batch_size, n_frames, max_n_events)

        if event_padding_mask is not None:
            event_padding_mask = event_padding_mask.unsqueeze(1).repeat(1, n_frames, 1)
            attn = attn.masked_fill(event_padding_mask == 0, float('-inf'))

        out = torch.softmax(attn, dim=-1)

        assert not out.isnan().any()
        return out
    

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


    @classmethod
    def from_config(cls, config: ModelConfig):
        return cls(**config._asdict())


    def forward(self,
                audio_frames: Tensor,
                score_ids: LongTensor,
                score_attn_mask: BoolTensor,
                score_to_event: BoolTensor,
                event_padding_mask: BoolTensor,
                **_) -> Tensor:
        """
        Args:
            audio_frames (Tensor): Mel spectrogram frames.
                    Size: (batch_size, n_frames, d_audio).
            score_ids (LongTensor): Integer input ids for the score encoder.
                    Size: (batch_size, max_n_tokens, d_score)
            score_attn_mask (BoolTensor): Boolean self-attention mask for the
                    score encoder. Size: (batch_size, max_n_tokens, max_n_tokens).
            score_to_event (BoolTensor): TODO
            event_padding_mask (BoolTensor): TODO

        Returns:
            Tensor: Alignment matrix.
                    Size: (batch_size, n_frames, max_n_events)
        """
        score_embed: Tensor = self.ScoreEncoder(score_ids, score_attn_mask)
        event_embed = score_to_event.float() @ score_embed
        audio_embed: Tensor = self.AudioEncoder(audio_frames,
                                                 event_embed,
                                                 event_padding_mask)
        out = self.Head(audio_embed, event_embed, event_padding_mask)

        assert not out.isnan().any()
        return out
