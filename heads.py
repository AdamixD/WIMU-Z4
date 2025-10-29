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
