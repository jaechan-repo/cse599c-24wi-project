import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor, LongTensor
from typing import Optional
from ..utils.constants import *


class MultiheadALiBiSelfAttention(nn.Module):

    def __init__(self, d_embed, n_heads, dropout: float = 0):
        super(MultiheadALiBiSelfAttention, self).__init__()

        assert d_embed % n_heads == 0

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.fc_q = nn.Linear(d_embed, d_embed)
        self.fc_k = nn.Linear(d_embed, d_embed)
        self.fc_v = nn.Linear(d_embed, d_embed)
        self.fc_o = nn.Linear(d_embed, d_embed)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_head]))


    def forward(self, embed: Tensor,
                attn_mask: Optional[BoolTensor] = None
                ) -> Tensor:
        bsz, n_embed, d = embed.shape
        assert d == self.d_embed

        Q: Tensor = self.fc_q(embed)
        K: Tensor = self.fc_k(embed)
        V: Tensor = self.fc_v(embed)

        ### ALiBi: BEGIN ###
        r = torch.arange(n_embed)
        pos = torch.abs(r.unsqueeze(0) - r.unsqueeze(1))
        step = 8 / self.n_heads
        ms = torch.pow(2, -torch.arange(step, 8+step, step)).view(-1, 1, 1)
        ALiBi = (ms * pos.unsqueeze(0).repeat(self.n_heads, 1, 1)).unsqueeze(0)
        ### ALiBi: END ###

        Q_h = Q.view(bsz, -1, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K_h = K.view(bsz, -1, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        assert Q_h.shape == K_h.shape == (bsz, self.n_heads, n_embed, self.d_head)

        attn = Q_h @ K_h.permute(0, 1, 3, 2)
        attn = (attn - ALiBi) / self.scale
        assert attn.shape == (bsz, self.n_heads, n_embed, n_embed)

        if attn_mask is not None:
            attn_mask = attn_mask.float().unsqueeze(1)
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        attn = self.dropout(torch.softmax(attn, dim=-1))
        V_h = V.view(bsz, -1, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        out = torch.matmul(attn, V_h)
        assert out.shape == (bsz, self.n_heads, n_embed, self.d_head)

        out = out.permute(0, 2, 1, 3).contiguous()
        assert out.shape == (bsz, out, self.n_heads, self.d_head)

        out = out.view(bsz, -1, self.d_embed)
        out = self.fc_o(out)
        assert out.shape == embed.shape

        return out


class ScoreEncoderBlock(nn.Module):

    def __init__(self, d_embed, n_head,
                 attn_dropout=0.0, ffn_dropout=0.5,
                 ffn_expansion=4):
        super(ScoreEncoderBlock, self).__init__()

        self.attn = MultiheadALiBiSelfAttention(d_embed, n_head, attn_dropout)

        self.norm1 = nn.LayerNorm(d_embed)
        self.norm2 = nn.LayerNorm(d_embed)

        self.ffn = nn.Sequential(
            nn.Linear(d_embed, ffn_expansion * d_embed),
            nn.ReLU(),
            nn.Linear(ffn_expansion * d_embed, d_embed),
        )

        self.dropout = nn.Dropout(ffn_dropout)


    def forward(self, embed: Tensor,
                attn_mask: Optional[Tensor] = None
                ) -> Tensor:
        out1 = self.attn(embed, attn_mask)
        out1 = self.dropout(self.norm1(out1 + embed))

        out2 = self.ffn(out1)
        out2 = self.dropout(self.norm2(out2 + out1))
        assert out2.shape == embed.shape
        return out2


class ScoreEncoder(nn.Module):

    def __init__(self,
                 vocab_size, d_embed, n_head,
                 attn_dropout=0.0, ffn_dropout=0.5,
                 ffn_expansion=4, n_layers=5):
        super(ScoreEncoder, self).__init__()

        self.lookup_embed = nn.Embedding(vocab_size, d_embed,
                                         padding_idx=TOKEN_ID['[PAD]'])
        self.encoder_blocks = nn.ModuleList([
            ScoreEncoderBlock(
                d_embed, n_head, attn_dropout, ffn_dropout, ffn_expansion
            ) for _ in range(n_layers)
        ])


    def forward(self,
                input_ids: BoolTensor,
                attn_mask: Optional[BoolTensor] = None
                ) -> Tensor:
        embed: Tensor = self.lookup_embed(input_ids)
        for block in self.encoder_blocks:
            embed = block(embed, attn_mask)
        return embed
  
  
class AudioEncoderBlock(nn.Module):

    def __init__(self, d_audio, d_score, n_heads,
                 attn_dropout=0.0,
                 ffn_dropout=0.5, ffn_expansion=4):
        super(AudioEncoderBlock, self).__init__()

        self.fc_q = nn.Linear(d_audio, d_score)
        self.fc_k = nn.Linear(d_score, d_score)
        self.fc_v = nn.Linear(d_score, d_audio)

        self.self_attn = MultiheadALiBiSelfAttention(d_audio, n_heads, attn_dropout)
        self.xattn = nn.MultiheadAttention(d_audio, n_heads, attn_dropout,
                                           batch_first=True)

        self.norm1 = nn.LayerNorm(d_audio)
        self.norm2 = nn.LayerNorm(d_audio)
        self.ffn = nn.Sequential(
            nn.Linear(d_audio, ffn_expansion * d_audio),
            nn.ReLU(),
            nn.Linear(ffn_expansion * d_audio, d_audio),
        )
        self.dropout = nn.Dropout(ffn_dropout)


    def forward(self, 
                audio_embed: Tensor, 
                event_embed: Tensor,
                event_padding_mask: Optional[BoolTensor] = None):
        """
        Args:
            audio_embed (Tensor): Audio embedding
            event_embed (Tensor): Score embedding
            event_padding_mask (Optional[BoolTensor]):
        """
        Q: Tensor = self.fc_q(audio_embed)
        K: Tensor = self.fc_k(event_embed)
        V: Tensor = self.fc_v(event_embed)

        out1 = self.self_attn(audio_embed)
        out1 = self.xattn(Q, K, V,
                          key_padding_mask=event_padding_mask,
                          need_weights=False)
        out1 = self.dropout(self.norm1(out1 + audio_embed))
        assert out1.shape == audio_embed.shape

        out2 = self.ffn(out1)
        out2 = self.dropout(self.norm2(out2 + out1))
        assert out2.shape == out1.shape
        return out2
    

class AudioEncoder(nn.Module):

    def __init__(self, d_audio, d_score, n_heads,
                 attn_dropout=0.0,
                 ffn_dropout=0.5,
                 ffn_expansion=4,
                 n_layers=5):
        super(AudioEncoder, self).__init__()

        self.encoder_blocks = nn.ModuleList([AudioEncoderBlock(
            d_audio, d_score, n_heads,
            attn_dropout, ffn_dropout, ffn_expansion
        ) for _ in range(n_layers)])


    def forward(self,
                audio_frames: Tensor,
                event_embed: Tensor,
                event_padding_mask: Optional[BoolTensor] = None
                ) -> Tensor:
        out = audio_frames
        for block in self.encoder_blocks:
            out = block(out, event_embed, event_padding_mask)
        return out


class CrossAttentionHead(nn.Module):

    def __init__(self, d_audio, d_score):
        super().__init__()
        self.fc_q = nn.Linear(d_audio, d_score)
        self.fc_k = nn.Linear(d_score, d_score)

    def forward(self,
                audio_embed: Tensor,
                event_embed: Tensor,
                event_padding_mask: Optional[BoolTensor] = None
                ) -> Tensor:
        
        bsz, n_frames, _ = audio_embed.shape
        _, max_n_events, _ = event_embed.shape

        Q: Tensor = self.fc_q(audio_embed)
        K: Tensor = self.fc_k(event_embed)
        attn = Q @ K.transpose(1, 2)
        assert(attn.shape == (bsz, n_frames, max_n_events))

        if event_padding_mask is not None:
            attn = attn.masked_fill(event_padding_mask == 0, float('-inf'))

        return torch.softmax(attn, dim=-1)
    

class AlignerModel(nn.Module):

    def __init__(self, vocab_size, d_score, n_heads_score,
                 d_audio, n_heads_audio, 
                 attn_dropout_score=0.0, ffn_dropout_score=0.5, n_layers_score=5,
                 attn_dropout_audio=0.0, ffn_dropout_audio=0.5, n_layers_audio=5):

        super(AlignerModel, self).__init__()
        self.score_encoder = ScoreEncoder(vocab_size, d_score, n_heads_score,
                                          attn_dropout_score, ffn_dropout_score,
                                          n_layers_score)
        self.audio_encoder = AudioEncoder(d_audio, d_score, n_heads_audio,
                                          attn_dropout_audio, ffn_dropout_audio,
                                          n_layers_audio)
        self.head = CrossAttentionHead(d_audio, d_score)


    def forward(self,
                audio_frames: Tensor,
                score_ids: LongTensor,
                score_attn_mask: BoolTensor,
                score_to_events: BoolTensor,
                event_padding_mask: BoolTensor,
                ) -> Tensor:
        score_embed: Tensor = self.score_encoder(score_ids, score_attn_mask)
        event_embed = score_to_events.float() @ score_embed
        audio_embed: Tensor = self.audio_encoder(
                audio_frames, event_embed, event_padding_mask)
        out = self.head(audio_embed, event_embed, event_padding_mask)
        return out
