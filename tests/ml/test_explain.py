"""SHAP structure and non-constant contributions. Trains tiny fixture models — not BDG2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

shap = pytest.importorskip("shap")

from ml.config import apply_smoke_overrides, load_config
from ml.explain.compute import linear_coefficient_shap, shap_tabular
from ml.explain.payload import local_payload
from ml.explain.run import run_explanations
from ml.models.tabular import LinearRegressionForecaster, RandomForestForecaster
from ml.train import run_training

# Spec-example numbers must never appear as a canned SHAP payload.
_SPEC_EXAMPLE_CONTRIB = 3.8
_SPEC_EXAMPLE_PRED = 47.8


def _train(tmp_path: Path, fixture_frame, names: list[str]) -> dict:
    config = apply_smoke_overrides(load_config())
    return run_training(
        frame=fixture_frame,
        config=config,
        artifacts_dir=tmp_path,
        model_names=names,
        smoke=True,
        force_ratio=True,
        data_source="synthetic_fixture",
        persist_db=False,
    )


def _explain(tmp_path: Path, fixture_frame, names: list[str], *, local_max: int = 4) -> dict:
    config = apply_smoke_overrides(load_config())
    return run_explanations(
        artifacts_dir=tmp_path,
        frame=fixture_frame,
        config=config,
        model_names=names,
        data_source="synthetic_fixture",
        data_label="TEST FIXTURE",
        background_max=32,
        local_max=local_max,
        sequence_max_evals=120,
        permutation_repeats=1,
        random_seed=42,
        write_plots=True,
        persist_db=False,
        force_ratio=True,
    )


def test_local_payload_structure_from_explainer_not_literals():
    rng = np.random.default_rng(0)
    names = ["air_temperature", "lag_1", "occupancy"]
    contribs = rng.normal(size=3)
    values = rng.normal(size=3)
    pred = float(10.0 + contribs.sum())
    payload = local_payload(
        prediction=pred,
        feature_names=names,
        feature_values=values,
        contributions=contribs,
        model_name="linear_regression",
        explainer="LinearExplainer",
        occupancy_source="simulated",
        data_label="TEST FIXTURE",
    )
    assert set(payload.keys()) >= {"prediction", "features", "model_name", "explainer"}
    assert payload["features"][0].keys() >= {"name", "value", "contribution"}
    assert payload["prediction"] == pytest.approx(pred)
    assert payload["features"][0]["contribution"] == pytest.approx(float(contribs[0]))
    assert "simulated occupancy" in payload["narrative"]
    assert "caused" not in payload["narrative"].lower()


def test_linear_coefficient_shap_matches_model_not_constants():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3))
    # y depends on column 0 only
    y = 2.5 * x[:, 0] - 1.0
    model = LinearRegressionForecaster()
    model.fit(x[:30], y[:30], feature_names=["a", "b", "c"])
    shap_values, base = linear_coefficient_shap(model.model, x[30:], x[:30])
    assert shap_values.shape == (10, 3)
    assert np.mean(np.abs(shap_values[:, 0])) > np.mean(np.abs(shap_values[:, 1]))
    assert not np.allclose(shap_values, _SPEC_EXAMPLE_CONTRIB)
    pred = model.predict(x[30:])
    np.testing.assert_allclose(pred, base + shap_values.sum(axis=1), rtol=1e-6, atol=1e-6)


def test_tree_and_linear_shap_on_fixture_are_computed(fixture_frame, tmp_path: Path):
    names = ["linear_regression", "random_forest"]
    _train(tmp_path, fixture_frame, names)
    manifest = _explain(tmp_path, fixture_frame, names, local_max=4)
    assert manifest["data_label"] == "TEST FIXTURE"
    assert manifest["occupancy_source"] == "simulated"

    for name in names:
        global_path = tmp_path / "explanations" / name / "global.json"
        local_path = tmp_path / "explanations" / name / "local.json"
        assert global_path.exists()
        assert local_path.exists()
        import json

        global_payload = json.loads(global_path.read_text(encoding="utf-8"))
        locals_payload = json.loads(local_path.read_text(encoding="utf-8"))
        assert global_payload["ranking"]
        assert all("mean_abs_shap" in row for row in global_payload["ranking"])
        scores = [row["mean_abs_shap"] for row in global_payload["ranking"]]
        assert scores == sorted(scores, reverse=True)
        assert not all(s == _SPEC_EXAMPLE_CONTRIB for s in scores)
        assert global_payload["explainer"] in {"LinearExplainer", "LinearExplainer_coefficients", "TreeExplainer"}
        assert len(locals_payload) >= 2
        first, second = locals_payload[0], locals_payload[1]
        for item in (first, second):
            assert "prediction" in item and "features" in item
            assert item["occupancy_source"] == "simulated"
            assert item["data_label"] == "TEST FIXTURE"
            assert item["prediction"] != pytest.approx(_SPEC_EXAMPLE_PRED)
            names_in = [f["name"] for f in item["features"]]
            contribs = np.array([f["contribution"] for f in item["features"]], dtype=np.float64)
            assert "occupancy" in names_in
            assert contribs.shape[0] == len(names_in)
            assert not np.allclose(contribs, contribs[0])
            assert not np.allclose(contribs, _SPEC_EXAMPLE_CONTRIB)
            occ = next(f for f in item["features"] if f["name"] == "occupancy")
            if abs(occ["contribution"]) > 0:
                assert "measured occupancy" not in item["narrative"].lower()
        c0 = np.array([f["contribution"] for f in first["features"]], dtype=np.float64)
        c1 = np.array([f["contribution"] for f in second["features"]], dtype=np.float64)
        assert not np.allclose(c0, c1), "two rows produced identical SHAP — possible hardcoded array"
        if first["expected_value"] is not None:
            pred_hat = first["expected_value"] + float(c0.sum())
            np.testing.assert_allclose(first["prediction"], pred_hat, rtol=1e-3, atol=1e-3)


def test_xgboost_treeshap_not_constant(fixture_frame, tmp_path: Path):
    _train(tmp_path, fixture_frame, ["xgboost"])
    _explain(tmp_path, fixture_frame, ["xgboost"], local_max=3)
    import json

    locals_payload = json.loads((tmp_path / "explanations" / "xgboost" / "local.json").read_text(encoding="utf-8"))
    global_payload = json.loads((tmp_path / "explanations" / "xgboost" / "global.json").read_text(encoding="utf-8"))
    assert global_payload["explainer"] == "TreeExplainer"
    contribs = np.array([f["contribution"] for f in locals_payload[0]["features"]], dtype=np.float64)
    assert contribs.std() > 0


def test_shap_tabular_linear_explainer_interface():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 4))
    y = x[:, 0] - 0.5 * x[:, 1]
    model = LinearRegressionForecaster()
    model.fit(x[:40], y[:40], feature_names=list("abcd"))
    values, base, used = shap_tabular(
        model_name="linear_regression",
        sklearn_model=model.model,
        X=x[40:],
        background=x[:40],
    )
    assert used in {"LinearExplainer", "LinearExplainer_coefficients"}
    assert values.shape == (10, 4)
    pred = model.predict(x[40:])
    np.testing.assert_allclose(pred, base + values.sum(axis=1), rtol=1e-5, atol=1e-5)


def test_shap_tabular_tree_explainer_interface():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 3))
    y = np.where(x[:, 0] > 0, 10.0, 1.0) + 0.1 * rng.normal(size=80)
    model = RandomForestForecaster(config={"n_estimators": 8, "max_depth": 4, "n_jobs": 1}, random_seed=0)
    model.fit(x[:60], y[:60], feature_names=["a", "b", "c"])
    values, base, used = shap_tabular(
        model_name="random_forest",
        sklearn_model=model.model,
        X=x[60:],
        background=x[:60],
    )
    assert used == "TreeExplainer"
    assert values.shape == (20, 3)
    assert np.mean(np.abs(values[:, 0])) >= np.mean(np.abs(values[:, 2])) * 0.5
