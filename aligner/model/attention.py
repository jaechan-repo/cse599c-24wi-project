import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from typing import Optional, Literal
from ..utils.constants import *
import math
from .pos_encoding import RotaryEmbedding


class MultiheadAttention(nn.Module):
    """Uses rotary embeddings.
    """
    
    def __init__(self,
                 d_embed: int,
                 d_k: int,
                 n_heads: int,
                 bias: bool = True,
                 dropout: float = 0.0):
        super().__init__()

        assert d_k % n_heads == 0, \
                "Attention dim must be divisible by specified num heads."

        self.d_embed, self.d_k = d_embed, d_k
        self.n_heads = n_heads
        self.d_h = d_embed // n_heads

        self.Dropout = nn.Dropout(dropout)

        self.RotaryEmbedding = RotaryEmbedding(self.d_h // 2)
        self.Rotate = lambda x: self.RotaryEmbedding.rotate_queries_or_keys(x)

        self.Linear_Q = nn.Linear(d_embed, d_embed, bias=bias)
        self.Linear_K = nn.Linear(d_k, d_embed, bias=bias)

        self.scale = 1 / math.sqrt(self.d_embed)
        self.Linear_V = nn.Linear(d_k, d_embed, bias=bias)
        self.Linear_O = nn.Linear(d_embed, d_embed, bias=bias)


    def forward(self,
                embed: Tensor,
                key: Tensor,
                attn_mask: BoolTensor | None = None
                ) -> Tensor:
        bsz, len_q, d_embed = embed.shape
        # assert d_embed == self.d_embed

        _, len_k, _ = key.shape
        # assert d_k == self.d_k

        Q, K, V = self.Linear_Q(embed), self.Linear_K(key), self.Linear_V(key)
        Q_h = Q.view(bsz, len_q, self.n_heads, self.d_h).permute(0, 2, 1, 3)
        K_h = K.view(bsz, len_k, self.n_heads, self.d_h).permute(0, 2, 1, 3)
        V_h = V.view(bsz, len_k, self.n_heads, self.d_h).permute(0, 2, 1, 3)
        # (bsz, n_heads, seq_len, d_h)

        Q_h, K_h = self.Rotate(Q_h), self.Rotate(K_h)
        attn = Q_h @ K_h.transpose(2, 3)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            attn = attn.masked_fill(~attn_mask.to(attn.device), float('-inf'))

        attn *= self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.Dropout(attn)
        out = attn @ V_h
        # assert out.shape == (bsz, self.n_heads, len_q, self.d_h)

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(bsz, len_q, d_embed)
        out: Tensor = self.Linear_O(out)

        assert not out.isnan().any()
        return out


class MultiheadSelfAttention(nn.Module):

    def __init__(self,
                 d_embed: int,
                 n_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        self.Attn = MultiheadAttention(d_embed,
                                       d_embed,
                                       n_heads,
                                       dropout=dropout)

    def forward(self,
                embed: Tensor,
                attn_mask: Optional[Tensor] = None
                ) -> Tensor:
        out = self.Attn(embed, embed, attn_mask)
        assert not out.isnan().any()
        return out
