import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Literal
from ..utils.constants import *
import math


NEG_INF = -1e9


class MultiheadAttention(nn.Module):
    
    def __init__(self,
                 d_embed: int,
                 d_k: int,
                 n_heads: int,
                 bias: bool = True,
                 dropout: float = 0.0,
                 pos_encoding: Literal["alibi"] | None = None):
        super(MultiheadAttention, self).__init__()

        assert d_k % n_heads == 0, \
                "Attention dim must be divisible by specified num heads."

        self.d_embed, self.d_k = d_embed, d_k
        self.n_heads = n_heads
        self.d_h = d_embed // n_heads

        self.Dropout = nn.Dropout(dropout)
        self.Linear_Q = nn.Linear(d_embed, d_embed, bias=bias)
        self.Linear_K = nn.Linear(d_k, d_embed, bias=bias)
        self.Linear_V = nn.Linear(d_k, d_embed, bias=bias)
        self.Linear_O = nn.Linear(d_embed, d_embed, bias=bias)

        self.scale = 1 / math.sqrt(self.d_embed)
        self.pos_encoding = pos_encoding


    def forward(self,
                embed: Tensor,
                key: Tensor,
                attn_mask: Optional[Tensor] = None
                ) -> Tensor:

        bsz, len_q, d_embed = embed.shape
        # assert d_embed == self.d_embed

        _, len_k, d_k = key.shape
        # assert d_k == self.d_k

        Q, K, V = self.Linear_Q(embed), self.Linear_K(key), self.Linear_V(key)
        Q_h = Q.view(bsz, len_q, self.n_heads, self.d_h).permute(0, 2, 1, 3)
        K_h = K.view(bsz, len_k, self.n_heads, self.d_h).permute(0, 2, 1, 3)
        V_h = V.view(bsz, len_k, self.n_heads, self.d_h).permute(0, 2, 1, 3)

        attn = Q_h @ K_h.transpose(2, 3)

        if self.pos_encoding == 'alibi':
            assert len_q == len_k, "ALiBi is only supported for self-attention."
            r = torch.arange(len_q)
            pos = torch.abs(r.unsqueeze(0) - r.unsqueeze(1))
            step = 8 / self.n_heads
            ms = torch.pow(2, -torch.arange(step, 8+step, step)).view(-1, 1, 1)
            ALiBi = (ms * pos.unsqueeze(0).repeat(self.n_heads, 1, 1)).unsqueeze(0)
            attn -= ALiBi.type_as(attn)

        attn *= self.scale

        if attn_mask is not None:
            attn_mask = attn_mask.float().unsqueeze(1)
            attn = attn.masked_fill(attn_mask == 0, NEG_INF)

        attn = self.Dropout(torch.softmax(attn, dim=-1))

        out = attn @ V_h
        # assert out.shape == (bsz, self.n_heads, len_q, self.d_h)

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(bsz, len_q, d_embed)
        out = self.Linear_O(out)

        assert not out.isnan().any()
        return out


class MultiheadALiBiSelfAttention(nn.Module):

    def __init__(self,
                 d_embed: int,
                 n_heads: int,
                 dropout: float = 0.0):
        super(MultiheadALiBiSelfAttention, self).__init__()
        self.Attn = MultiheadAttention(d_embed,
                                       d_embed,
                                       n_heads,
                                       dropout=dropout,
                                       pos_encoding='alibi')

    def forward(self,
                embed: Tensor,
                attn_mask: Optional[Tensor] = None
                ) -> Tensor:
        out = self.Attn(embed, embed, attn_mask)
        assert not out.isnan().any()
        return out
