import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MultiHeadAlibiSelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, hid_dim, n_heads, dropout=0, device="cuda"):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(input_dim, hid_dim).to(device)
        self.fc_k = nn.Linear(input_dim, hid_dim).to(device)
        self.fc_v = nn.Linear(input_dim, hid_dim).to(device)

        self.fc_o = nn.Linear(hid_dim, hid_dim).to(device)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)


    def forward(self, input, padding_mask = None):
        """
          input: [bsz, seq_len, input_dim]
          padding_mask: [bsz, seq_len]
        """

        batch_size = input.shape[0]
        len_k = input.shape[1]
        len_q = input.shape[1]
        len_v = input.shape[1]

        query = self.fc_q(input)
        key = self.fc_k(input)
        value = self.fc_v(input)

        r = torch.arange(len_k)
        pos = torch.abs(r.unsqueeze(0) - r.unsqueeze(1))

        step = 8/self.n_heads
        ms = torch.pow(2, -torch.arange(step, 8+step, step)).view(-1, 1, 1)

        alibi = (ms * pos.unsqueeze(0).repeat(self.n_heads, 1, 1)).unsqueeze(0).cuda()

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # bsz, n_heads, l, head_dim
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # bsz, n_heads, l, head_dim
        attn = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) # bsz, n_heads, l, l. the attention matrix

        attn = (attn - alibi) / self.scale

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).cuda() # B 1 l l
            attn = attn.masked_fill(padding_mask == 0, float('-inf'))

        attn = self.dropout(torch.softmax(attn, dim = -1))

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

class CrossAttentionAlignerLayer (nn.Module):
  def __init__(self, hid_dim, n_heads, dropout=0.5, device="cuda"):
    super().__init__()
    self.q_emb = nn.Linear(hid_dim, hid_dim)
    self.k_emb = nn.Linear(hid_dim, hid_dim)
    self.v_emb = nn.Linear(hid_dim, hid_dim)
    self.x_attention = nn.MultiheadAttention(hid_dim, n_heads, dropout, batch_first=True)

  def forward(self, score_embs, audio_embs, x_attn_mask=None):
    """
      score_embs: (bsz, max_num_events, hid_dim)
      audio_embs: (bsz, audio_length, hid_dim)
      x_attn_mask: (bsz, max_num_events)
    """
    query = self.q_emb(audio_embs)
    key = self.k_emb(score_embs)
    value = self.v_emb(score_embs)

    output = self.x_attention(query, key, value, x_attn_mask)[0]
    return output

class ScoreAlignerLayer (nn.Module):
  def __init__(self, vocab_size, hid_dim, n_heads, dropout=0.5, num_layers=5, device="cuda"):
      super(ScoreAlignerLayer, self).__init__()
      # print("OK2")
      self.embed = nn.Linear(vocab_size, hid_dim).to(device)
      self.self_attn = [MultiHeadAlibiSelfAttentionLayer(input_dim=hid_dim, hid_dim=hid_dim, n_heads=n_heads, dropout=dropout, device=device).to(device) for i in range(num_layers)]
      self.num_layers = num_layers

  def forward(self, input, score_mask=None):
      """
        input: [B, score_length, vocab_size], one hots
        score_mask: [B, score_length, score_length]
      """

      embs = self.embed(input)
      tmp = embs
      for i in range(self.num_layers):
        tmp = self.self_attn[i](tmp, score_mask)


      return tmp

class SpectrogramAlignerLayer(nn.Module):
  def __init__(self, input_dim, hid_dim, n_heads, dropout=0.5, num_layers=5, device="cuda"): # score representation
      super(SpectrogramAlignerLayer, self).__init__()
      self.num_layers = num_layers
      # print("***", num_layers)
      # print(input_dim, hid_dim, n_heads, dropout, device, num_layers)
      self.self_attn = [MultiHeadAlibiSelfAttentionLayer(input_dim=input_dim, hid_dim=hid_dim, n_heads=n_heads, dropout=dropout, device=device).to(device) for i in range(num_layers)]
      self.x_attn = [CrossAttentionAlignerLayer(hid_dim, n_heads, dropout).to(device) for i in range(num_layers)]
      self.num_layers = num_layers
      # print(self.self_attn)
      # print("OK")

  def forward(self, audio_embs, score_embs, x_attn_mask=None):
      """
        audio_embs: [B, num_frames, input_dim]
        score_embs: [B, max_num_events, hid]
        x_attn_mask: [B, max_num_events]
      """
      tmp = audio_embs
      # print("*****")
      # print(score_embs)
      for i in range(self.num_layers):
          # print("+", i)
          tmp = self.self_attn[i](tmp)
          # print("-", i, x_attn_mask.type(), tmp.shape)
          tmp = self.x_attn[i](score_embs, tmp, x_attn_mask)
          # print('/', i, tmp)

      return tmp

class CrossAttnMatrix(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, audio_embs, score_embs, x_attn_mask=None):
    """
      audio_embs: (bsz, num_frames, dim)
      score_embs: (bsz, max_num_events, dim)
      x_attn_mask: (bsz, max_num_events)
    """
    attn = torch.matmul(audio_embs, score_embs.permute(0, 2, 1)) # (bsz, num_frames, max_num_events)
    if x_attn_mask is not None:
      attn = attn.masked_fill(x_attn_mask == 0, float('-inf'))
    return torch.softmax(attn, dim=-1)

class TheAligner(nn.Module):
  def __init__(self, input_dim_audio, hid_dim_audio, vocab_size, hid_dim_score, n_heads_audio, n_heads_score,
               dropout_audio=0.5, dropout_score=0.5, num_layers_audio=5, num_layers_score=5, device="cuda"):
    super().__init__()
    self.spectAligner = SpectrogramAlignerLayer(input_dim_audio, hid_dim_audio, n_heads_audio, dropout_audio, num_layers_audio, device)
    self.scoreAligner = ScoreAlignerLayer(vocab_size, hid_dim_score, n_heads_score, dropout_score, num_layers_score, device)
    self.proj = nn.Linear(hid_dim_score, hid_dim_audio).to(device)
    self.cross = CrossAttnMatrix().to(device)
    self.device = device

  def forward(self, audio_embs, score_embs, score_mask, x_attn_mask, event_proj):
    """
      audio_embs: [bsz, num_frames, input_dim_audio]
      score_embs: [bsz, score_len, vocab_size] (one hot)
      score_mask: [bsz, score_len, score_len]
      x_attn_mask: [bsz, max_num_events]
      event_proj: [score_len, max_num_events]
    """
    score_embs = self.scoreAligner(score_embs, score_mask) # [bsz, score_len, hid_dim_score]
    print("score_embs:", score_embs, score_embs.shape)
    score_embs_events = torch.matmul(score_embs.permute(0, 2, 1), event_proj).permute(0, 2, 1).to(self.device) # [bsz, max_num_events, hid_dim_score]
    print(score_embs_events, score_embs_events.shape)
    score_embs = self.proj(score_embs_events) # [bsz, max_num_events, hid_dim_audio]
    spect_embs = self.spectAligner(audio_embs, score_embs, x_attn_mask) # [bsz, num_frames, hid_dim_audio] 
    # print("spect_embs", spect_embs, spect_embs.shape)
    # print("score_embs", score_embs, score_embs.shape)
    output = self.cross(audio_embs, score_embs, x_attn_mask) # [bsz, num_frames, max_num_events]
    return output
