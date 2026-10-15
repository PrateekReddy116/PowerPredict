"""Linear Regression, Random Forest, and XGBoost. Tabular Phase 2 feature matrix → y_{t+1}."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from ml.models.base import as_1d, ensure_parent


class _TabularMixin:
    name: str
    model: Any
    feature_names: list[str] | None
    config: dict[str, Any]
    random_seed: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        return as_1d(self.model.predict(np.asarray(X, dtype=np.float64)))

    def save(self, path: Path) -> None:
        ensure_parent(path)
        joblib.dump(
            {
                "name": self.name,
                "model": self.model,
                "feature_names": self.feature_names,
                "config": self.config,
                "random_seed": self.random_seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Any:
        payload = joblib.load(path)
        obj = cls(config=payload.get("config") or {}, random_seed=payload.get("random_seed", 42))
        obj.model = payload["model"]
        obj.feature_names = payload.get("feature_names")
        return obj


class LinearRegressionForecaster(_TabularMixin):
    name = "linear_regression"

    def __init__(self, config: dict[str, Any] | None = None, random_seed: int = 42) -> None:
        self.config = dict(config or {})
        self.random_seed = int(random_seed)
        self.feature_names: list[str] | None = None
        self.model = LinearRegression()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        del X_val, y_val
        self.feature_names = feature_names
        self.model.fit(np.asarray(X_train, dtype=np.float64), as_1d(y_train))


class RandomForestForecaster(_TabularMixin):
    name = "random_forest"

    def __init__(self, config: dict[str, Any] | None = None, random_seed: int = 42) -> None:
        self.config = dict(config or {})
        self.random_seed = int(random_seed)
        self.feature_names: list[str] | None = None
        self.model = RandomForestRegressor(
            n_estimators=int(self.config.get("n_estimators", 100)),
            max_depth=self.config.get("max_depth", 12),
            min_samples_leaf=int(self.config.get("min_samples_leaf", 2)),
            n_jobs=int(self.config.get("n_jobs", 1)),
            random_state=self.random_seed,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        del X_val, y_val
        self.feature_names = feature_names
        self.model.fit(np.asarray(X_train, dtype=np.float64), as_1d(y_train))


class XGBoostForecaster(_TabularMixin):
    name = "xgboost"

    def __init__(self, config: dict[str, Any] | None = None, random_seed: int = 42) -> None:
        self.config = dict(config or {})
        self.random_seed = int(random_seed)
        self.feature_names: list[str] | None = None
        self.model = XGBRegressor(
            n_estimators=int(self.config.get("n_estimators", 200)),
            max_depth=int(self.config.get("max_depth", 6)),
            learning_rate=float(self.config.get("learning_rate", 0.05)),
            subsample=float(self.config.get("subsample", 0.8)),
            colsample_bytree=float(self.config.get("colsample_bytree", 0.8)),
            n_jobs=int(self.config.get("n_jobs", 1)),
            random_state=self.random_seed,
            objective="reg:squarederror",
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.feature_names = feature_names
        x_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = as_1d(y_train)
        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        early = self.config.get("early_stopping_rounds")
        if has_val and early is not None:
            self.model.set_params(early_stopping_rounds=int(early))
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if has_val:
            fit_kwargs["eval_set"] = [(np.asarray(X_val, dtype=np.float64), as_1d(y_val))]
        try:
            self.model.fit(x_tr, y_tr, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("verbose", None)
            self.model.fit(x_tr, y_tr, **fit_kwargs)
