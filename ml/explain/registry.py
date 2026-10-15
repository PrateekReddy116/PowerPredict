"""Explainer choice per model. KernelSHAP is never used on an HTTP request."""

from __future__ import annotations

from typing import Any

# Live = fast enough to compute for a single row (CLI still precomputes; the API loads files).
# Offline = test-subset only, written under artifacts/.../explanations/.
EXPLAINER_SPECS: dict[str, dict[str, Any]] = {
    "linear_regression": {
        "explainer": "LinearExplainer",
        "mode": "live",
        "rationale": (
            "Linear models have a closed-form interventional SHAP: "
            "phi_i = coef_i * (x_i - E[x_i]). shap.LinearExplainer (or the equivalent "
            "coefficient formula) is exact and millisecond-fast."
        ),
    },
    "random_forest": {
        "explainer": "TreeExplainer",
        "mode": "live",
        "rationale": (
            "TreeSHAP (shap.TreeExplainer) computes polynomial-time exact Shapley values "
            "for ensembles of trees. Suitable for interactive use; not KernelSHAP."
        ),
    },
    "xgboost": {
        "explainer": "TreeExplainer",
        "mode": "live",
        "rationale": (
            "Same TreeSHAP algorithm on the XGBoost ensemble. Exact for the tree model class "
            "and fast enough for a single prediction."
        ),
    },
    "lstm": {
        "explainer": "PermutationExplainer",
        "mode": "offline",
        "rationale": (
            "LSTM is not a tree or linear model. KernelSHAP on every HTTP request is too slow "
            "and would not make the network 'more explainable' at serving time. Offline "
            "shap.PermutationExplainer on a test subset (sequence attributions summed over time) "
            "is stored in artifacts. Permutation importance is the fallback if PermutationExplainer "
            "cannot run."
        ),
    },
    "cnn_lstm": {
        "explainer": "PermutationExplainer",
        "mode": "offline",
        "rationale": (
            "Same policy as LSTM. CNN-LSTM is not claimed to be more explainable via live "
            "KernelSHAP-per-request. Offline permutation SHAP (or permutation importance) only."
        ),
    },
}


def explainer_name(model_name: str) -> str:
    spec = EXPLAINER_SPECS.get(model_name)
    if spec is None:
        raise KeyError(f"no explainer registered for model {model_name!r}")
    return str(spec["explainer"])


def explainer_mode(model_name: str) -> str:
    spec = EXPLAINER_SPECS.get(model_name)
    if spec is None:
        raise KeyError(f"no explainer registered for model {model_name!r}")
    return str(spec["mode"])


def explainer_spec(model_name: str) -> dict[str, Any]:
    if model_name not in EXPLAINER_SPECS:
        raise KeyError(f"no explainer registered for model {model_name!r}")
    return dict(EXPLAINER_SPECS[model_name])


def is_live_explainer(model_name: str) -> bool:
    return explainer_mode(model_name) == "live"
