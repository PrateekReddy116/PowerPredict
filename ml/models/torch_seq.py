"""Shared PyTorch sequence training: CPU, early stopping on val RMSE, no DataLoader shuffle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.models.base import as_1d, ensure_parent


class _SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        self.y = torch.as_tensor(as_1d(y).astype(np.float32), dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    ds = _SeqDataset(X, y)
    bs = max(1, min(int(batch_size), len(ds)))
    # shuffle=False: keep chronological batch order. Splits are already chronological.
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)


def _predict_scaled(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    if len(X) == 0:
        return np.empty((0,), dtype=np.float64)
    loader = _loader(X, np.zeros(len(X)), batch_size)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            out = model(xb.to(device))
            chunks.append(out.detach().cpu().numpy().reshape(-1))
    return np.concatenate(chunks).astype(np.float64)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = as_1d(y_true), as_1d(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def inverse_target(scaler: StandardScaler | None, y: np.ndarray) -> np.ndarray:
    if scaler is None:
        return as_1d(y)
    return scaler.inverse_transform(as_1d(y).reshape(-1, 1)).reshape(-1)


class TorchSequenceForecaster:
    """Base for LSTM / CNN-LSTM. Fit scalers/checkpoints on train/val only."""

    name: str
    net: nn.Module

    def __init__(self, config: dict[str, Any] | None = None, random_seed: int = 42) -> None:
        self.config = dict(config or {})
        self.random_seed = int(random_seed)
        self.device = torch.device(self.config.get("device", "cpu"))
        self.target_scaler: StandardScaler | None = None
        self.seq_len = int(self.config.get("seq_len", 24))
        self.n_features: int | None = None
        self.best_val_rmse: float | None = None
        self.epochs_trained: int = 0

    def _build_net(self, n_features: int) -> nn.Module:
        raise NotImplementedError

    def _arch_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        if X_train.ndim != 3:
            raise ValueError("sequence models expect X with shape (n, seq_len, n_features)")
        self.n_features = int(X_train.shape[-1])
        self.seq_len = int(X_train.shape[1])
        self.net = self._build_net(self.n_features).to(self.device)

        y_tr = as_1d(y_train)
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(y_tr.reshape(-1, 1))
        y_tr_s = self.target_scaler.transform(y_tr.reshape(-1, 1)).reshape(-1)

        y_va_s = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            y_va_s = self.target_scaler.transform(as_1d(y_val).reshape(-1, 1)).reshape(-1)

        epochs = int(self.config.get("epochs", 30))
        patience = int(self.config.get("patience", 5))
        batch_size = int(self.config.get("batch_size", 32))
        lr = float(self.config.get("learning_rate", 1e-3))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        train_loader = _loader(X_train, y_tr_s, batch_size)

        best_rmse = float("inf")
        best_state: dict[str, Any] | None = None
        stale = 0
        trained = 0
        for _epoch in range(epochs):
            self.net.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
            trained += 1
            if y_va_s is None or X_val is None:
                continue
            val_pred_s = _predict_scaled(self.net, X_val, self.device, batch_size)
            val_pred = inverse_target(self.target_scaler, val_pred_s)
            val_rmse = _rmse(y_val, val_pred)
            if val_rmse + 1e-12 < best_rmse:
                best_rmse = val_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        self.epochs_trained = trained
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.best_val_rmse = best_rmse
        elif y_va_s is None:
            self.best_val_rmse = None
        else:
            val_pred_s = _predict_scaled(self.net, X_val, self.device, batch_size)
            self.best_val_rmse = _rmse(y_val, inverse_target(self.target_scaler, val_pred_s))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "net"):
            raise RuntimeError("model is not fitted")
        batch_size = int(self.config.get("batch_size", 32))
        pred_s = _predict_scaled(self.net, X, self.device, batch_size)
        return inverse_target(self.target_scaler, pred_s)

    def save(self, path: Path) -> None:
        if not hasattr(self, "net"):
            raise RuntimeError("model is not fitted")
        ensure_parent(path)
        target = None
        if self.target_scaler is not None:
            target = {
                "mean_": np.asarray(self.target_scaler.mean_),
                "scale_": np.asarray(self.target_scaler.scale_),
                "var_": np.asarray(self.target_scaler.var_),
                "n_features_in_": int(self.target_scaler.n_features_in_),
            }
        torch.save(
            {
                "name": self.name,
                "state_dict": self.net.state_dict(),
                "arch": self._arch_kwargs(),
                "n_features": self.n_features,
                "seq_len": self.seq_len,
                "config": self.config,
                "random_seed": self.random_seed,
                "target_scaler": target,
                "best_val_rmse": self.best_val_rmse,
                "epochs_trained": self.epochs_trained,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TorchSequenceForecaster:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        cfg = dict(payload.get("config") or {})
        obj = cls(config=cfg, random_seed=payload.get("random_seed", 42))
        obj.n_features = int(payload["n_features"])
        obj.seq_len = int(payload.get("seq_len", cfg.get("seq_len", 24)))
        obj.net = obj._build_net(obj.n_features).to(obj.device)
        obj.net.load_state_dict(payload["state_dict"])
        obj.net.eval()
        obj.best_val_rmse = payload.get("best_val_rmse")
        obj.epochs_trained = int(payload.get("epochs_trained") or 0)
        ts = payload.get("target_scaler")
        if ts is not None:
            scaler = StandardScaler()
            scaler.mean_ = np.asarray(ts["mean_"])
            scaler.scale_ = np.asarray(ts["scale_"])
            scaler.var_ = np.asarray(ts["var_"])
            scaler.n_features_in_ = int(ts["n_features_in_"])
            obj.target_scaler = scaler
        return obj
