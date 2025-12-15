import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerNextNoteModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, nlayers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.vel_proj = nn.Linear(1, d_model)
        self.dur_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.pitch_head = nn.Linear(d_model, vocab_size)
        self.vel_head = nn.Linear(d_model, 1)
        self.dur_head = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, tokens, velocities=None, durations=None):
        b, seq_len = tokens.shape
        x = self.token_embed(tokens) * math.sqrt(self.d_model)
        if velocities is not None:
            v = velocities.unsqueeze(-1)
            x = x + self.vel_proj(v)
        if durations is not None:
            d = durations.unsqueeze(-1)
            x = x + self.dur_proj(d)
        x = self.pos_enc(x)
        x = self.transformer(x)
        last = x[:, -1, :]
        pitch_logits = self.pitch_head(last)
        vel_pred = self.vel_head(last).squeeze(-1)
        dur_pred = self.dur_head(last).squeeze(-1)
        return pitch_logits, vel_pred, dur_pred
