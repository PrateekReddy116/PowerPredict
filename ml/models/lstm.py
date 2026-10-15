"""LSTM next-hour forecaster.

Architecture (CPU-sized):
    Input (batch, seq_len, n_features)
    -> LSTM(hidden_size≈64, num_layers=1–2, batch_first=True)
    -> last hidden state
    -> optional dropout (used when num_layers==1; LSTM's own dropout needs >1 layer)
    -> Linear(hidden_size, 1) next-hour kWh (in scaled target space during training)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ml.models.torch_seq import TorchSequenceForecaster


class LSTMNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head_dropout = nn.Dropout(dropout) if num_layers == 1 and dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.head(self.head_dropout(last))


class LSTMForecaster(TorchSequenceForecaster):
    name = "lstm"

    def _arch_kwargs(self) -> dict[str, Any]:
        return {
            "hidden_size": int(self.config.get("hidden_size", 64)),
            "num_layers": int(self.config.get("num_layers", 1)),
            "dropout": float(self.config.get("dropout", 0.1)),
        }

    def _build_net(self, n_features: int) -> nn.Module:
        kw = self._arch_kwargs()
        return LSTMNet(n_features=n_features, **kw)
