"""Offline SHAP explanations. The API loads saved JSON; it does not run KernelSHAP."""

from ml.explain.narrative import (
    narrative_from_contributions,
    narrative_from_mean_abs,
    phrase_for_feature,
)
from ml.explain.payload import global_payload, local_payload
from ml.explain.registry import EXPLAINER_SPECS, explainer_mode, explainer_name, explainer_spec
from ml.explain.run import run_explanations

__all__ = [
    "EXPLAINER_SPECS",
    "explainer_mode",
    "explainer_name",
    "explainer_spec",
    "global_payload",
    "local_payload",
    "narrative_from_contributions",
    "narrative_from_mean_abs",
    "phrase_for_feature",
    "run_explanations",
]
