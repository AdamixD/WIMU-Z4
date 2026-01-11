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


class CNNLSTMHead(nn.Module):
    """
    CNN + LSTM head for VA regression.

    Based on https://www.researchgate.net/publication/383237611_MUSIC_EMOTION_RECOGNITION_AND_CLASSIFICATION_USING_HYBRID_CNN-LSTM_DEEP_NEURAL_NETWORK
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.4,
        kernel_size: int = 9,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.post_lstm_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, 2)
        self.act = nn.Tanh()

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.lstm(z)
        z = self.post_lstm_dropout(z)
        z = self.out(z)
        return self.act(z)


class CNNLSTMClassificationHead(nn.Module):
    """CNN + LSTM head for Russell4Q classification."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.4,
        kernel_size: int = 9,
        num_classes: int = 4,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.post_lstm_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.lstm(z)
        z = self.post_lstm_dropout(z)
        return self.out(z)
