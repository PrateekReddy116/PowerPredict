"""CNN-LSTM next-hour forecaster.

Architecture (CPU-sized):
    Input (batch, seq_len, n_features)
    -> permute to (batch, n_features, seq_len) for Conv1d
    -> Conv1d(out_channels≈32, kernel=3, padding=1) + ReLU
    -> MaxPool1d(kernel=2)  (sequence length ≈ seq_len/2)
    -> permute to (batch, seq_pooled, conv_channels)
    -> LSTM(hidden_size≈64, num_layers=1–2, batch_first=True)
    -> last hidden state
    -> Linear(hidden_size, 1) next-hour kWh (scaled target during training)

Superiority vs LSTM or tabular models is not assumed; compare test metrics.json.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ml.models.torch_seq import TorchSequenceForecaster


class CNNLSTMNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head_dropout = nn.Dropout(dropout) if num_layers == 1 and dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        conv_in = x.transpose(1, 2)
        conv_out = torch.relu(self.conv(conv_in))
        pooled = self.pool(conv_out)
        seq = pooled.transpose(1, 2)
        output, _ = self.lstm(seq)
        last = output[:, -1, :]
        return self.head(self.head_dropout(last))


class CNNLSTMForecaster(TorchSequenceForecaster):
    name = "cnn_lstm"

    def _arch_kwargs(self) -> dict[str, Any]:
        return {
            "conv_channels": int(self.config.get("conv_channels", 32)),
            "kernel_size": int(self.config.get("kernel_size", 3)),
            "hidden_size": int(self.config.get("hidden_size", 64)),
            "num_layers": int(self.config.get("num_layers", 1)),
            "dropout": float(self.config.get("dropout", 0.1)),
        }

    def _build_net(self, n_features: int) -> nn.Module:
        kw = self._arch_kwargs()
        return CNNLSTMNet(n_features=n_features, **kw)
