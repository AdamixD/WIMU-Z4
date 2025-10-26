from __future__ import annotations

import torch.nn as nn


class BiGRUHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        z, _ = self.gru(x)
        z = self.drop(z)
        return self.out(z)


class TemporalConvHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, k: int = 3, levels: int = 3, dropout: float = 0.1):
        super().__init__()
        ch = hidden
        layers = []
        d = 1
        for i in range(levels):
            layers += [
                nn.Conv1d(in_dim if i == 0 else ch, ch, k, padding=d * (k - 1) // 2, dilation=d),
                nn.ReLU(),
                nn.Conv1d(ch, ch, k, padding=d * (k - 1) // 2, dilation=d),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            d *= 2
        self.net = nn.Sequential(*layers)
        self.out = nn.Conv1d(ch, 2, 1)

    def forward(self, x):
        z = self.net(x.transpose(1, 2))
        return self.out(z).transpose(1, 2)


class TransformerHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, nhead: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.out = nn.Linear(hidden, 2)

    def forward(self, x):
        z = self.proj(x)
        z = self.enc(z)
        return self.out(z)
