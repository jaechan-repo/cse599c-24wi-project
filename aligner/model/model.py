import torch
import torch.nn as nn
from torch import Tensor, BoolTensor, LongTensor, NamedTuple
from typing import Optional
from ..utils.constants import *


class ModelConfig(NamedTuple):
    vocab_size: int = max(TOKEN_ID) + 1,
    d_score: int = 64,
    n_heads_score: int = 4

    d_audio: int = N_MELS

    n_heads_score: int = 4
    attn_dropout_score: float = 0.0
    ffn_dropout_score: float = 0.5
    n_layers_score: int = 5

    n_heads_audio: int = 8
    attn_dropout_audio: float = 0.0
    ffn_dropout_audio: float = 0.5
    n_layers_audio: int = 5


class MultiheadALiBiSelfAttention(nn.Module):

    def __init__(self,
                 d_embed: int,
                 n_heads: int,
                 dropout: float = 0.0,
                 d_k: Optional[int] = None):
        """
        Args:
            d_embed (int): Input embedding dimension.
            n_heads (int): Number of attention heads.
            d_k (Optional[int]): Key dimension. 
                    Defaults to d_embed if not specified.
            dropout (float, optional): Probability of attention dropout.
                    Defaults to 0.
        """
        super(MultiheadALiBiSelfAttention, self).__init__()

        assert d_embed % n_heads == 0

        if d_k is not None:
            d_k = d_embed

        self.d_embed = d_embed  # 64
        self.d_k = d_k
        self.n_heads = n_heads  # 4

        self.d_kh = d_k / n_heads
        self.d_vh = d_embed / n_heads

        self.fc_q = nn.Linear(d_embed, d_k)
        self.fc_k = nn.Linear(d_embed, d_k)
        self.fc_v = nn.Linear(d_embed, d_embed)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.Tensor([self.d_kh]))


    def forward(self,
                embed: Tensor,
                attn_mask: Optional[BoolTensor] = None
                ) -> Tensor:
        """
        Args:
            embed (Tensor): Input embeddings. Size: (batch_size, seq_len, d_embed)
            attn_mask (Optional[BoolTensor]): Boolean attention mask.
                    Size: (batch_size, seq_len, seq_len)

        Returns:
            Tensor: Contextualized embeddings of the same size as the input.
        """
        batch_size, seq_len, d_embed = embed.shape
        assert d_embed == self.d_embed

        Q: Tensor = self.fc_q(embed)    # (batch_size, seq_len, d_k)
        K: Tensor = self.fc_k(embed)    # (batch_size, seq_len, d_k)
        V: Tensor = self.fc_v(embed)    # (batch_size, seq_len, d_v)

        ############################################
        ################ ALiBi: BEGIN ##############
        ############################################
        r = torch.arange(seq_len)
        pos = torch.abs(r.unsqueeze(0) - r.unsqueeze(1))
        step = 8 / self.n_heads
        ms = torch.pow(2, -torch.arange(step, 8+step, step)).view(-1, 1, 1)
        ALiBi = (ms * pos.unsqueeze(0).repeat(self.n_heads, 1, 1)).unsqueeze(0)
        ############################################
        ################# ALiBi: END ###############
        ############################################

        Q_h = Q.view(batch_size, -1, self.n_heads, self.d_hk).permute(0, 2, 1, 3)
        K_h = K.view(batch_size, -1, self.n_heads, self.d_hk).permute(0, 2, 1, 3)
        assert Q_h.shape == K_h.shape == (batch_size, self.n_heads, seq_len, self.d_hk)

        # Scaled dot product
        attn = Q_h @ K_h.transpose(2, 3)
        attn -= ALiBi
        attn /= self.scale
        assert attn.shape == (batch_size, self.n_heads, seq_len, seq_len)

        if attn_mask is not None:
            attn_mask = attn_mask.float().unsqueeze(1)
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        # Each query embedding receives a probability distribution
        attn = self.dropout(torch.softmax(attn, dim=-1))
        V_h = V.view(batch_size, seq_len, self.n_heads, self.d_vh).permute(0, 2, 1, 3)

        out = attn @ V_h
        assert out.shape == (batch_size, self.n_heads, seq_len, self.d_vh)

        out = out.permute(0, 2, 1, 3).contiguous()
        assert out.shape == (batch_size, out, self.n_heads, self.d_vh)

        out = out.view(batch_size, seq_len, self.d_embed)
        out = self.fc_o(out)
        assert out.shape == embed.shape

        return out


class ScoreEncoderBlock(nn.Module):

    def __init__(self,
                 d_embed: int,
                 n_heads: int,
                 attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.5,
                 ffn_expansion: int = 4):
        super(ScoreEncoderBlock, self).__init__()

        self.self_attn = MultiheadALiBiSelfAttention(d_embed, n_heads, attn_dropout)

        self.norm1 = nn.LayerNorm(d_embed)
        self.norm2 = nn.LayerNorm(d_embed)

        self.ffn = nn.Sequential(
            nn.Linear(d_embed, ffn_expansion * d_embed),
            nn.ReLU(),
            nn.Linear(ffn_expansion * d_embed, d_embed),
        )

        self.dropout = nn.Dropout(ffn_dropout)


    def forward(self,
                embed: Tensor,
                attn_mask: Optional[Tensor] = None
                ) -> Tensor:
        out1 = self.self_attn(embed, attn_mask)
        out1 = self.dropout(self.norm1(out1 + embed))

        out2 = self.ffn(out1)
        out2 = self.dropout(self.norm2(out2 + out1))
        assert out2.shape == embed.shape
        return out2


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

        self.lookup_embed = nn.Embedding(vocab_size, d_score,
                                         padding_idx=TOKEN_ID['[PAD]'])

        self.encoder_blocks = nn.ModuleList([
            ScoreEncoderBlock(
                d_score, n_heads, attn_dropout, ffn_dropout, ffn_expansion
            ) for _ in range(n_layers)
        ])


    def forward(self,
                input_ids: LongTensor,
                attn_mask: Optional[BoolTensor] = None
                ) -> Tensor:
        embed: Tensor = self.lookup_embed(input_ids)
        for block in self.encoder_blocks:
            embed = block(embed, attn_mask)
        return embed
  

class AudioEncoderBlock(nn.Module):

    def __init__(self,
                 d_audio: int,
                 d_score: int,
                 n_heads: int,
                 attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.5,
                 ffn_expansion: int = 4):
        super(AudioEncoderBlock, self).__init__()

        self.self_attn = MultiheadALiBiSelfAttention(d_audio, n_heads, attn_dropout)

        self.fc_q = nn.Linear(d_audio, d_score)     # for cross-attention
        self.fc_k = nn.Linear(d_score, d_score)     # for cross-attention
        self.fc_v = nn.Linear(d_score, d_audio)     # for cross-attention
        self.xattn = nn.MultiheadAttention(d_audio, n_heads, attn_dropout,
                                           batch_first=True)

        self.norm1 = nn.LayerNorm(d_audio)
        self.norm2 = nn.LayerNorm(d_audio)
        self.norm3 = nn.LayerNorm(d_audio)
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

        ### Step 1: Self-attention ###
        out1 = self.self_attn(audio_embed)
        out1 = self.dropout(self.norm1(out1 + audio_embed)) # residual
        assert out1.shape == audio_embed.shape
        
        ### Step 2: Cross-attention ###
        Q_audio: Tensor = self.fc_q(out1)
        K_event: Tensor = self.fc_k(event_embed)
        V_event: Tensor = self.fc_v(event_embed)
        out2 = self.xattn(Q_audio, K_event, V_event,
                    key_padding_mask=event_padding_mask,
                    need_weights=False)
        out2 = self.dropout(self.norm2(out2 + out1))
        assert out2.shape == out1.shape

        ### Step 3: Feedforward network ###
        out3 = self.ffn(out2)
        out3 = self.dropout(self.norm3(out3 + out2))
        assert out3.shape == out2.shape

        return out3
    

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
    """Alignment head implemented as cross-attention.
    The only learnable components are the query and key projectors.
    """

    def __init__(self, d_audio: int, d_score: int):
        super().__init__()
        self.fc_q = nn.Linear(d_audio, d_score)
        self.fc_k = nn.Linear(d_score, d_score)


    def forward(self,
                audio_embed: Tensor,
                event_embed: Tensor,
                event_padding_mask: Optional[BoolTensor] = None
                ) -> Tensor:
        
        batch_size, n_frames, _ = audio_embed.shape
        _, max_n_events, _ = event_embed.shape

        Q: Tensor = self.fc_q(audio_embed)
        K: Tensor = self.fc_k(event_embed)
        attn = Q @ K.transpose(1, 2)
        assert attn.shape == (batch_size, n_frames, max_n_events)

        if event_padding_mask is not None:
            attn = attn.masked_fill(event_padding_mask == 0, float('-inf'))

        return torch.softmax(attn, dim=-1)
    

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
                 n_layers_audio: int = 5):
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
        self.score_encoder = ScoreEncoder(vocab_size, d_score, n_heads_score,
                                          attn_dropout_score, ffn_dropout_score,
                                          n_layers_score)
        self.audio_encoder = AudioEncoder(d_audio, d_score, n_heads_audio,
                                          attn_dropout_audio, ffn_dropout_audio,
                                          n_layers_audio)
        self.head = CrossAttentionHead(d_audio, d_score)


    @staticmethod
    def _proj_score_to_event(score_embed: Tensor,
                             event_padding_mask: BoolTensor
                             ) -> Tensor:
        with torch.no_grad():
            num_ones = event_padding_mask.sum(dim=1)
            max_ones = num_ones.max()
            score_to_event = torch.zeros(event_padding_mask.size(0),
                                        max_ones.item(),
                                        event_padding_mask.size(1),
                                        dtype=torch.bool)
            for i in range(event_padding_mask.size(0)):
                indices = torch.where(event_padding_mask[i])[0]
                score_to_event[i, torch.arange(num_ones[i].item()), indices] = True

        event_embed = score_to_event.float() @ score_embed
        return event_embed


    def forward(self,
                audio_frames: Tensor,
                score_ids: LongTensor,
                score_attn_mask: BoolTensor,
                event_padding_mask: BoolTensor,
                ) -> Tensor:
        """
        Args:
            audio_frames (Tensor): Mel spectrogram frames.
                    Size: (batch_size, n_frames, d_audio).
            score_ids (LongTensor): Integer input ids for the score encoder.
                    Size: (batch_size, max_n_tokens, d_score)
            score_attn_mask (BoolTensor): Boolean self-attention mask for the
                    score encoder. Size: (batch_size, max_n_tokens, max_n_tokens).
            event_padding_mask (BoolTensor): Boolean tensor that specifies
                    where in the score tokens (`score_ids`) the event markers occur.
                    Size: (batch_size, max_n_tokens).

        Returns:
            Tensor: Alignment matrix.
                    Size: (batch_size, n_frames, max_n_events)
        """
        score_embed: Tensor = self.score_encoder(score_ids, score_attn_mask)
        event_embed = AlignerModel._proj_score_to_event(score_embed,
                                                        event_padding_mask)
        audio_embed: Tensor = self.audio_encoder(audio_frames,
                                                 event_embed,
                                                 event_padding_mask)
        out = self.head(audio_embed, event_embed, event_padding_mask)
        return out
