"""SHAP summary / importance plots from computed arrays — never from hardcoded ranks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_mean_abs_bar(
    feature_names: list[str],
    mean_abs: np.ndarray,
    path: Path,
    *,
    title: str,
    max_display: int = 20,
) -> Path:
    _ensure_parent(path)
    scores = np.asarray(mean_abs, dtype=np.float64).reshape(-1)
    names = list(feature_names)
    order = np.argsort(scores)[::-1][:max_display]
    names_o = [names[i] for i in order][::-1]
    scores_o = scores[order][::-1]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(names_o) + 1)))
    ax.barh(names_o, scores_o, color="#c4a574")
    ax.set_xlabel("mean |SHAP| (kWh)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_shap_summary(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    path: Path,
    *,
    title: str,
) -> Path | None:
    """Beeswarm via shap.summary_plot. Returns None if shap plotting is unavailable."""
    _ensure_parent(path)
    try:
        import shap
    except ImportError:
        return None
    plt.close("all")
    kwargs = dict(
        features=np.asarray(features, dtype=np.float64),
        feature_names=list(feature_names),
        show=False,
        max_display=min(20, len(feature_names)),
    )
    try:
        shap.summary_plot(np.asarray(shap_values, dtype=np.float64), rng=0, **kwargs)
    except TypeError:
        shap.summary_plot(np.asarray(shap_values, dtype=np.float64), **kwargs)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return path
