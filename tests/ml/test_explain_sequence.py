"""Offline permutation SHAP for sequence models. Fixture only — not BDG2, not KernelSHAP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("shap")

from ml.config import apply_smoke_overrides, load_config
from ml.explain.run import run_explanations
from ml.train import run_training


def test_lstm_offline_shap_writes_payload(fixture_frame, tmp_path: Path):
    config = apply_smoke_overrides(load_config())
    run_training(
        frame=fixture_frame,
        config=config,
        artifacts_dir=tmp_path,
        model_names=["lstm"],
        smoke=True,
        force_ratio=True,
        data_source="synthetic_fixture",
        persist_db=False,
    )
    manifest = run_explanations(
        artifacts_dir=tmp_path,
        frame=fixture_frame,
        config=config,
        model_names=["lstm"],
        data_source="synthetic_fixture",
        data_label="TEST FIXTURE",
        background_max=8,
        local_max=2,
        sequence_max_evals=80,
        permutation_repeats=1,
        random_seed=42,
        write_plots=False,
        persist_db=False,
        force_ratio=True,
    )
    assert manifest["models"][0]["explainer"] in {"PermutationExplainer", "permutation_ablation"}
    local = json.loads((tmp_path / "explanations" / "lstm" / "local.json").read_text(encoding="utf-8"))
    assert len(local) >= 1
    item = local[0]
    assert item["explainer_mode"] == "offline"
    contribs = np.array([f["contribution"] for f in item["features"]], dtype=np.float64)
    assert contribs.shape[0] == len(item["features"])
    assert np.isfinite(contribs).all()
    assert item["prediction"] != pytest.approx(47.8)
    assert not np.allclose(contribs, 3.8)
