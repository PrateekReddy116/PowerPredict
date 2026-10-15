"""Compute SHAP (or permutation SHAP) from fitted models. Never returns hardcoded arrays."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ml.explain.registry import explainer_name, explainer_spec
from ml.models import is_sequence_model

LOGGER = logging.getLogger(__name__)


def _as_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 3 and array.shape[-1] == 1:
        return array.reshape(array.shape[0], array.shape[1])
    if array.ndim != 2:
        raise ValueError(f"expected SHAP matrix (n, n_features), got shape {array.shape}")
    return array


def _base_as_1d(base: Any, n_rows: int) -> np.ndarray:
    if base is None:
        return np.full(n_rows, np.nan, dtype=np.float64)
    array = np.asarray(base, dtype=np.float64).reshape(-1)
    if array.size == 1:
        return np.full(n_rows, float(array[0]), dtype=np.float64)
    if array.size == n_rows:
        return array
    return np.full(n_rows, float(np.mean(array)), dtype=np.float64)


def _from_explanation(result: Any, n_rows: int, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(result, "values"):
        values = np.asarray(result.values, dtype=np.float64)
        base = getattr(result, "base_values", None)
    else:
        values = np.asarray(result, dtype=np.float64)
        base = None
    if values.ndim == 3:
        # (n, features, outputs) or (n, seq, features) — last axis of size 1 is an output dim
        if values.shape[-1] == 1:
            values = values[..., 0]
        elif values.shape[1] != n_features and values.shape[-1] == n_features:
            values = values
        else:
            values = values.reshape(values.shape[0], -1)
    values = _as_2d(values)
    if values.shape[1] != n_features and values.size == n_rows * n_features:
        values = values.reshape(n_rows, n_features)
    if values.shape != (n_rows, n_features):
        raise ValueError(f"SHAP shape {values.shape} != ({n_rows}, {n_features})")
    return values, _base_as_1d(base, n_rows)


def linear_coefficient_shap(
    model: Any,
    X: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interventional SHAP for a linear model: phi = coef * (x - E[x]). Uses fitted coef_, not literals."""
    coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0])
    means = np.mean(np.asarray(background, dtype=np.float64), axis=0)
    x = np.asarray(X, dtype=np.float64)
    shap_values = (x - means) * coef
    expected = intercept + float(np.dot(means, coef))
    return shap_values, np.full(len(x), expected, dtype=np.float64)


def shap_tabular(
    *,
    model_name: str,
    sklearn_model: Any,
    X: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (shap_values, base_values, explainer_used)."""
    import shap

    x = np.asarray(X, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    n_rows, n_features = x.shape
    spec_name = explainer_name(model_name)

    if model_name == "linear_regression":
        try:
            explainer = shap.LinearExplainer(sklearn_model, bg)
            result = explainer(x)
            values, base = _from_explanation(result, n_rows, n_features)
            return values, base, "LinearExplainer"
        except Exception as exc:
            LOGGER.warning("LinearExplainer failed (%s); using coefficient SHAP", exc)
            values, base = linear_coefficient_shap(sklearn_model, x, bg)
            return values, base, "LinearExplainer_coefficients"

    if spec_name != "TreeExplainer":
        raise ValueError(f"{model_name} is not a tabular Tree/Linear explainer target")

    explainer = shap.TreeExplainer(sklearn_model)
    try:
        result = explainer(x)
        values, base = _from_explanation(result, n_rows, n_features)
    except Exception:
        raw = explainer.shap_values(x)
        values = _as_2d(raw)
        if values.shape != (n_rows, n_features):
            raise ValueError(f"TreeSHAP shape {values.shape} != ({n_rows}, {n_features})")
        expected = getattr(explainer, "expected_value", None)
        base = _base_as_1d(expected, n_rows)
    return values, base, "TreeExplainer"


def _predict_sequences(forecaster: Any, X_seq: np.ndarray) -> np.ndarray:
    pred = forecaster.predict(np.asarray(X_seq, dtype=np.float64))
    return np.asarray(pred, dtype=np.float64).reshape(-1)


def shap_sequences(
    *,
    model_name: str,
    forecaster: Any,
    X_seq: np.ndarray,
    background: np.ndarray,
    max_evals: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Permutation SHAP on flattened windows, then sum over time → per-feature contributions.

    KernelExplainer is not used. This is offline-only (callers must not invoke per HTTP request).
    """
    import shap

    x = np.asarray(X_seq, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    if x.ndim != 3 or bg.ndim != 3:
        raise ValueError("sequence SHAP expects X with shape (n, seq_len, n_features)")
    n, seq_len, n_features = x.shape
    n_flat = seq_len * n_features
    x_flat = x.reshape(n, n_flat)
    bg_flat = bg.reshape(len(bg), n_flat)
    min_evals = 2 * n_flat + 1
    evals = min_evals if max_evals is None else max(int(max_evals), min_evals)

    def _fn(flat: np.ndarray) -> np.ndarray:
        arr = np.asarray(flat, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return _predict_sequences(forecaster, arr.reshape(arr.shape[0], seq_len, n_features))

    explainer = shap.PermutationExplainer(_fn, bg_flat, silent=True)
    try:
        result = explainer(x_flat, max_evals=evals)
        flat_values, base = _from_explanation(result, n, n_flat)
        used = "PermutationExplainer"
    except Exception as exc:
        LOGGER.warning("PermutationExplainer failed (%s); using channel ablation", exc)
        flat_values, base = _channel_ablation(forecaster, x, bg)
        # ablation already returns (n, n_features)
        return flat_values, base, "permutation_ablation"

    aggregated = flat_values.reshape(n, seq_len, n_features).sum(axis=1)
    return aggregated, base, used


def _channel_ablation(
    forecaster: Any,
    X_seq: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace each feature channel with the background mean; delta vs f(x). Not exact Shapley."""
    x = np.asarray(X_seq, dtype=np.float64)
    bg_mean = np.mean(np.asarray(background, dtype=np.float64), axis=0, keepdims=True)
    preds = _predict_sequences(forecaster, x)
    n, _seq_len, n_features = x.shape
    values = np.zeros((n, n_features), dtype=np.float64)
    for j in range(n_features):
        xp = x.copy()
        xp[:, :, j] = bg_mean[:, :, j]
        values[:, j] = preds - _predict_sequences(forecaster, xp)
    return values, preds - values.sum(axis=1)


def permutation_importance_sequences(
    *,
    forecaster: Any,
    X_seq: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 3,
    random_seed: int = 42,
) -> np.ndarray:
    """sklearn-style permutation importance (RMSE increase). Used as a global ranking check for DL."""
    rng = np.random.default_rng(int(random_seed))
    x = np.asarray(X_seq, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = _predict_sequences(forecaster, x)
    baseline = float(np.sqrt(np.mean((y - pred) ** 2)))
    n, _seq, n_features = x.shape
    importances = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        deltas: list[float] = []
        for _ in range(max(1, int(n_repeats))):
            xp = x.copy()
            idx = rng.permutation(n)
            xp[:, :, j] = x[idx, :, j]
            pred_p = _predict_sequences(forecaster, xp)
            rmse_p = float(np.sqrt(np.mean((y - pred_p) ** 2)))
            deltas.append(rmse_p - baseline)
        importances[j] = float(np.mean(deltas))
    return importances


def subsample_indices(n: int, k: int, *, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    take = min(int(k), int(n))
    chosen = rng.choice(n, size=take, replace=False)
    return np.sort(chosen)


def last_timestep_unscaled(X_unscaled_seq: np.ndarray) -> np.ndarray:
    return np.asarray(X_unscaled_seq, dtype=np.float64)[:, -1, :]


def mean_abs(shap_values: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.asarray(shap_values, dtype=np.float64)), axis=0)


def explain_model(
    *,
    model_name: str,
    forecaster: Any,
    X_explain: np.ndarray,
    X_background: np.ndarray,
    y_explain: np.ndarray | None = None,
    sequence_max_evals: int | None = None,
    permutation_repeats: int = 2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Dispatch TreeSHAP / LinearExplainer / offline PermutationExplainer."""
    spec = explainer_spec(model_name)
    if is_sequence_model(model_name):
        shap_values, base, used = shap_sequences(
            model_name=model_name,
            forecaster=forecaster,
            X_seq=X_explain,
            background=X_background,
            max_evals=sequence_max_evals,
        )
        perm = None
        if y_explain is not None and len(y_explain) == len(X_explain):
            perm = permutation_importance_sequences(
                forecaster=forecaster,
                X_seq=X_explain,
                y=y_explain,
                n_repeats=permutation_repeats,
                random_seed=random_seed,
            )
        return {
            "shap_values": shap_values,
            "base_values": base,
            "explainer": used,
            "explainer_mode": spec["mode"],
            "rationale": spec["rationale"],
            "permutation_importance": perm,
        }

    sklearn_model = getattr(forecaster, "model", None)
    if sklearn_model is None:
        raise ValueError(f"{model_name} has no sklearn-compatible .model")
    shap_values, base, used = shap_tabular(
        model_name=model_name,
        sklearn_model=sklearn_model,
        X=X_explain,
        background=X_background,
    )
    return {
        "shap_values": shap_values,
        "base_values": base,
        "explainer": used,
        "explainer_mode": spec["mode"],
        "rationale": spec["rationale"],
        "permutation_importance": None,
    }
