import torch.nn as nn


class BiGRUHead(nn.Module):
    """BiGRU head for VA regression (outputs 2 continuous values)"""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, 2)
        self.act = nn.Tanh()

    def forward(self, x):
        z, _ = self.gru(x)
        z = self.drop(z)
        z = self.out(z)
        return self.act(z)


class BiGRUClassificationHead(nn.Module):
    """BiGRU head for Russell 4Q classification (outputs 4 class logits)"""

    def __init__(
        self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2, num_classes: int = 4
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, num_classes)
        # No activation here - raw logits for CrossEntropyLoss

    def forward(self, x):
        z, _ = self.gru(x)
        z = self.drop(z)
        z = self.out(z)
        return z  # Return logits (no softmax - done in loss function)
