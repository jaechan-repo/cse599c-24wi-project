import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MultiHeadAlibiSelfAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=0, device="cuda"):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim).to(device)
        self.fc_k = nn.Linear(hid_dim, hid_dim).to(device)
        self.fc_v = nn.Linear(hid_dim, hid_dim).to(device)

        self.fc_o = nn.Linear(hid_dim, hid_dim).to(device)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)


    def forward(self, input, key_padding_mask = None):
        #input = [batch size, seq len, hid dim]
        #key_padding_mask = [batch size, seq len]

        batch_size = input.shape[0]
        len_k = input.shape[1]
        len_q = input.shape[1]
        len_v = input.shape[1]

        query = self.fc_q(input)
        key = self.fc_k(input)
        value = self.fc_v(input)

        r = torch.arange(len_k)
        pos = torch.abs(r.unsqueeze(0) - r.unsqueeze(1))
        print(pos)
        step = 8/self.n_heads
        ms = torch.pow(2, -torch.arange(step, 8+step, step)).view(-1, 1, 1)

        alibi = (ms * pos.unsqueeze(0).repeat(self.n_heads, 1, 1)).unsqueeze(0).cuda()

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # bsz, n_heads, l, head_dim
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # bsz, n_heads, l, head_dim
        attn = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) # bsz, n_heads, l, l. the attention matrix
        print(key_padding_mask.shape)
        key_padding_mask = key_padding_mask.unsqueeze(-1).repeat(1, 1, key_padding_mask.shape[1]).permute(0, 2, 1).unsqueeze(1).cuda() #1 1 l l

        attn = (attn - alibi) / self.scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask == 0, float('-inf'))
        print(attn)
        attn = self.dropout(torch.softmax(attn, dim = -1))
        print(attn.shape)
        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)

        x = weight1

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        #x = [batch size, query len, hid dim]

        return x



class SpectrogramAlignerLayer(nn.Module):
  def __init__(self, hid_dim, n_heads, dropout, device="cuda"): # score representation
      super().__init__()
      self.self_attn = MultiHeadAlibiSelfAttentionLayer(hid_dim=hid_dim, n_heads=n_heads, dropout=dropout, device=device).to(device)

  def forward(self, input, key_padding_mask=None):
      return self.self_attn(input, key_padding_mask)


class ScoreAlignerLayer (nn.Module):
  def __init__(self, hid_dim, n_heads, vocab_size=130, dropout=0.5, device="cuda"):
      super().__init__()
      self.embed = nn.Linear(vocab_size, hid_dim).to(device) # separateh@dim=hid_dim, n_heads=n_heads, dropout=dropout, device=device).to(device)
      self.self_attn = MultiHeadAlibiSelfAttentionLayer(hid_dim=hid_dim, n_heads=n_heads, dropout=dropout, device=device).to(device)

  def forward(self, input, key_padding_mask=None):
      # input: [B, L, vocab_size], one hots
      embs = self.embed(input)
      print(embs)
      return self.self_attn(embs, key_padding_mask)


class CrossAttentionAlignerLayer (nn.Module): 
  def __init__(self, hid_dim, n_heads, dropout=0.5, device="cuda"):
    super().__init__()
    self.q_emb = nn.Linear(hid_dim, hid_dim)
    self.k_emb = nn.Linear(hid_dim, hid_dim)
    self.v_emb = nn.Linear(hid_dim, hid_dim)
    self.x_attention = nn.MultiheadAttention(hid_dim, n_heads, dropout, batch_first=True)

  def forward(self, score_embs, audio_embs, x_attn_mask):
    """
      score_embs: (B, score_length, hid_dim)
      audio_embs: (B, audio_length, hid_dim)
      attn_mask: (B, score_length), no mask for the audio
    """
    query = self.q_emb(audio_embs)
    key = self.k_emb(score_embs)
    value = self.v_emb(score_embs)

    output = self.x_attention(query, key, value, key_padding_mask = x_attn_mask)
    return output
    