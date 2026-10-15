from pathlib import Path

import numpy as np
import pytest
from ml.config import apply_smoke_overrides, load_config
from ml.models import ALL_MODEL_NAMES, FORECASTER_CLASSES, is_sequence_model
from ml.models.dataset import make_sequences_grouped, prepare_splits, scale_features


@pytest.fixture
def prepared_smoke(fixture_frame):
    config = apply_smoke_overrides(load_config())
    prepared = prepare_splits(fixture_frame, config, force_ratio=True)
    x_tr, x_va, x_te = scale_features(prepared)
    return prepared, x_tr, x_va, x_te, int(config["seq_len"])


@pytest.mark.parametrize("name", ALL_MODEL_NAMES)
def test_each_model_fit_predict_roundtrip(name, prepared_smoke, tmp_path: Path):
    prepared, x_tr, x_va, x_te, seq_len = prepared_smoke
    cls = FORECASTER_CLASSES[name]
    cfg = apply_smoke_overrides(load_config())["models"][name]
    cfg["seq_len"] = seq_len
    cfg["device"] = "cpu"
    model = cls(config=cfg, random_seed=42)

    if is_sequence_model(name):
        x_train, y_train, _, _ = make_sequences_grouped(
            x_tr, prepared.y_train, prepared.groups_train, seq_len
        )
        x_val, y_val, _, _ = make_sequences_grouped(
            x_va, prepared.y_val, prepared.groups_val, seq_len
        )
        x_test, _y_test, _, _ = make_sequences_grouped(
            x_te, prepared.y_test, prepared.groups_test, seq_len
        )
        model.fit(x_train, y_train, x_val, y_val)
        pred = model.predict(x_test)
        suffix = "model.pt"
    else:
        model.fit(x_tr, prepared.y_train, x_va, prepared.y_val, feature_names=prepared.feature_names)
        pred = model.predict(x_te)
        suffix = "model.joblib"

    assert pred.shape[0] > 0
    assert np.isfinite(pred).all()

    path = tmp_path / suffix
    model.save(path)
    loaded = cls.load(path)
    pred2 = loaded.predict(x_test if is_sequence_model(name) else x_te)
    np.testing.assert_allclose(pred, pred2, rtol=1e-5, atol=1e-5)
