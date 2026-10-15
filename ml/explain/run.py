"""Load a training run + feature matrix and write SHAP artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.explain.compute import explain_model, last_timestep_unscaled, mean_abs, subsample_indices
from ml.explain.payload import global_payload, local_payload
from ml.explain.persist import try_insert_explanation, write_manifest, write_model_explanations
from ml.explain.plots import plot_mean_abs_bar, plot_shap_summary
from ml.explain.registry import EXPLAINER_SPECS, explainer_spec
from ml.models import ALL_MODEL_NAMES, FORECASTER_CLASSES, is_sequence_model
from ml.models.dataset import (
    drop_incomplete,
    ensure_target,
    make_sequences_grouped,
    make_sequences_with_history_grouped,
    prepare_from_pre_split,
    prepare_splits,
    select_feature_columns,
    split_feature_frame,
)
from ml.models.base import json_safe, set_random_seeds

LOGGER = logging.getLogger("ml.explain")


def _iso_ts(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat()


def _meta_column(frame: pd.DataFrame, name: str, index: int) -> str | None:
    if name not in frame.columns:
        return None
    value = frame.iloc[index][name]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value)


def load_scaler_bundle(artifacts_dir: Path) -> dict[str, Any]:
    path = Path(artifacts_dir) / "feature_scaler.joblib"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; train first (`python -m ml.train`)")
    return joblib.load(path)


def load_forecaster(artifacts_dir: Path, name: str) -> Any:
    cls = FORECASTER_CLASSES[name]
    filename = "model.pt" if is_sequence_model(name) else "model.joblib"
    path = Path(artifacts_dir) / name / filename
    if not path.exists():
        raise FileNotFoundError(f"missing model artifact {path}")
    return cls.load(path)


def available_models(artifacts_dir: Path, names: list[str] | None = None) -> list[str]:
    wanted = list(names) if names else list(ALL_MODEL_NAMES)
    found: list[str] = []
    for name in wanted:
        filename = "model.pt" if is_sequence_model(name) else "model.joblib"
        if (Path(artifacts_dir) / name / filename).exists():
            found.append(name)
    return found


def _label_from_source(data_source: str) -> str:
    text = data_source.lower()
    if "fixture" in text or "synthetic" in text:
        return "TEST FIXTURE"
    if "bdg2" in text or "processed" in text:
        return "BDG2"
    return data_source


def _occupancy_source(frame: pd.DataFrame) -> str | None:
    if "occupancy_source" not in frame.columns:
        return "simulated" if "occupancy" in frame.columns else None
    values = frame["occupancy_source"].dropna().astype(str).unique()
    if len(values) == 1:
        return str(values[0])
    if len(values) > 1:
        return "simulated" if "simulated" in set(values) else str(values[0])
    return "simulated" if "occupancy" in frame.columns else None


def _data_origin(frame: pd.DataFrame) -> str | None:
    if "data_origin" not in frame.columns:
        return None
    values = frame["data_origin"].dropna().astype(str).unique()
    if len(values) == 0:
        return None
    return str(values[0]) if len(values) == 1 else ",".join(sorted(values))


def _test_frame_from_prepared(prepared) -> pd.DataFrame:
    """Rebuild a test-aligned frame for metadata. Prefer original columns on prepared arrays."""
    data: dict[str, Any] = {"ts": prepared.ts_test, "building_id": prepared.groups_test}
    return pd.DataFrame(data)


def prepare_for_explain(
    *,
    artifacts_dir: Path,
    frame: pd.DataFrame | None = None,
    train: pd.DataFrame | None = None,
    val: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    force_ratio: bool = False,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return prepared splits, scaled X, and the original test frame (unscaled features + meta)."""
    bundle = load_scaler_bundle(artifacts_dir)
    scaler = bundle["scaler"]
    feature_names: list[str] = list(bundle["feature_names"])
    if train is not None and val is not None and test is not None:
        prepared = prepare_from_pre_split(train, val, test)
        test_frame = drop_incomplete(ensure_target(test.copy()), feature_names).reset_index(drop=True)
    elif frame is not None:
        cfg = config or {}
        prepared = prepare_splits(frame, cfg, force_ratio=force_ratio)
        work = drop_incomplete(ensure_target(frame.copy()), select_feature_columns(ensure_target(frame)))
        # Recreate the same chronological test partition as prepare_splits.
        _tr, _va, test_part, _windows = split_feature_frame(work, cfg, force_ratio=force_ratio)
        test_frame = test_part.reset_index(drop=True)
    else:
        raise ValueError("provide frame or train/val/test")

    if list(prepared.feature_names) != feature_names:
        missing = [name for name in feature_names if name not in prepared.feature_names]
        extra = [name for name in prepared.feature_names if name not in feature_names]
        if missing:
            raise ValueError(f"feature matrix missing scaler columns: {missing}")
        # Reorder to the saved scaler contract; ignore extras.
        LOGGER.info("reordering features to saved scaler names (dropped extras=%s)", extra)

    order = [prepared.feature_names.index(name) for name in feature_names]

    def _scale(unscaled: np.ndarray) -> np.ndarray:
        return scaler.transform(unscaled[:, order])

    x_train_s = _scale(prepared.X_train)
    x_val_s = _scale(prepared.X_val)
    x_test_s = _scale(prepared.X_test)
    prepared.X_train = prepared.X_train[:, order]
    prepared.X_val = prepared.X_val[:, order]
    prepared.X_test = prepared.X_test[:, order]
    prepared.feature_names = feature_names
    prepared.scaler = scaler
    for name in feature_names:
        if name not in test_frame.columns:
            raise ValueError(f"test frame missing feature {name}")
    return prepared, x_train_s, x_val_s, x_test_s, test_frame


def _sequence_pack(
    prepared,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    seq_len: int,
) -> dict[str, np.ndarray]:
    x_tr, y_tr, ts_tr, _g_tr = make_sequences_grouped(
        x_train, prepared.y_train, prepared.groups_train, seq_len, ts=prepared.ts_train
    )
    x_hist = np.concatenate([x_train, x_val], axis=0)
    y_hist = np.concatenate([prepared.y_train, prepared.y_val], axis=0)
    g_hist = np.concatenate([prepared.groups_train, prepared.groups_val], axis=0)
    x_te, y_te, ts_te, groups_test_seq_arr = make_sequences_with_history_grouped(
        x_hist,
        y_hist,
        g_hist,
        x_test,
        prepared.y_test,
        prepared.groups_test,
        seq_len,
        ts=prepared.ts_test,
    )
    x_unscaled_seq, _, _, _ = make_sequences_with_history_grouped(
        np.concatenate([prepared.X_train, prepared.X_val], axis=0),
        y_hist,
        g_hist,
        prepared.X_test,
        prepared.y_test,
        prepared.groups_test,
        seq_len,
        ts=prepared.ts_test,
    )
    return {
        "X_train_seq": x_tr,
        "y_train_seq": y_tr,
        "ts_train_seq": ts_tr,
        "X_test_seq": x_te,
        "y_test_seq": y_te,
        "ts_test_seq": ts_te,
        "groups_test_seq": np.asarray(groups_test_seq_arr),
        "X_test_unscaled_seq": x_unscaled_seq,
    }


def run_explanations(
    *,
    artifacts_dir: Path,
    frame: pd.DataFrame | None = None,
    train: pd.DataFrame | None = None,
    val: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    model_names: list[str] | None = None,
    data_source: str = "unknown",
    data_label: str | None = None,
    background_max: int = 64,
    local_max: int = 8,
    sequence_max_evals: int | None = None,
    permutation_repeats: int = 2,
    random_seed: int = 42,
    write_plots: bool = True,
    persist_db: bool = True,
    force_ratio: bool = False,
) -> dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    set_random_seeds(int(random_seed))
    rng = np.random.default_rng(int(random_seed))
    label = data_label or _label_from_source(data_source)
    names = available_models(artifacts_dir, model_names)
    if not names:
        raise FileNotFoundError(f"no trained model artifacts under {artifacts_dir}")

    prepared, x_train_s, x_val_s, x_test_s, test_frame = prepare_for_explain(
        artifacts_dir=artifacts_dir,
        frame=frame,
        train=train,
        val=val,
        test=test,
        config=config,
        force_ratio=force_ratio,
    )
    feature_names = list(prepared.feature_names)
    occ = _occupancy_source(test_frame)
    origin = _data_origin(test_frame)
    bundle = load_scaler_bundle(artifacts_dir)
    seq_len = int(bundle.get("seq_len") or (config or {}).get("seq_len") or 24)

    bg_idx = subsample_indices(len(x_train_s), background_max, rng=rng)
    local_idx_tab = subsample_indices(len(x_test_s), local_max, rng=rng)

    seq_pack = None
    if any(is_sequence_model(name) for name in names):
        seq_pack = _sequence_pack(prepared, x_train_s, x_val_s, x_test_s, seq_len)

    models_out: list[dict[str, Any]] = []
    for name in names:
        LOGGER.info("explaining %s with %s", name, explainer_spec(name)["explainer"])
        model = load_forecaster(artifacts_dir, name)
        dest = artifacts_dir / "explanations" / name
        dest.mkdir(parents=True, exist_ok=True)

        if is_sequence_model(name):
            assert seq_pack is not None
            bg_seq_idx = subsample_indices(len(seq_pack["X_train_seq"]), background_max, rng=rng)
            loc_seq_idx = subsample_indices(len(seq_pack["X_test_seq"]), local_max, rng=rng)
            if len(loc_seq_idx) == 0 or len(bg_seq_idx) == 0:
                raise ValueError(f"{name}: not enough sequence windows to explain")
            x_exp = seq_pack["X_test_seq"][loc_seq_idx]
            x_bg = seq_pack["X_train_seq"][bg_seq_idx]
            y_exp = seq_pack["y_test_seq"][loc_seq_idx]
            result = explain_model(
                model_name=name,
                forecaster=model,
                X_explain=x_exp,
                X_background=x_bg,
                y_explain=y_exp,
                sequence_max_evals=sequence_max_evals,
                permutation_repeats=permutation_repeats,
                random_seed=random_seed,
            )
            preds = np.asarray(model.predict(x_exp), dtype=np.float64).reshape(-1)
            values_unscaled = last_timestep_unscaled(seq_pack["X_test_unscaled_seq"][loc_seq_idx])
            ts_local = seq_pack["ts_test_seq"][loc_seq_idx]
            groups_local = seq_pack["groups_test_seq"][loc_seq_idx]
            row_indices = loc_seq_idx
        else:
            if len(local_idx_tab) == 0 or len(bg_idx) == 0:
                raise ValueError(f"{name}: not enough tabular rows to explain")
            x_exp = x_test_s[local_idx_tab]
            x_bg = x_train_s[bg_idx]
            result = explain_model(
                model_name=name,
                forecaster=model,
                X_explain=x_exp,
                X_background=x_bg,
                random_seed=random_seed,
            )
            preds = np.asarray(model.predict(x_exp), dtype=np.float64).reshape(-1)
            # unscaled values: prepared.X_test is in prepared.feature_names order (now scaler order)
            values_unscaled = prepared.X_test[local_idx_tab]
            ts_local = prepared.ts_test[local_idx_tab]
            groups_local = prepared.groups_test[local_idx_tab]
            row_indices = local_idx_tab

        shap_values = np.asarray(result["shap_values"], dtype=np.float64)
        base_values = np.asarray(result["base_values"], dtype=np.float64)
        used = str(result["explainer"])
        abs_mean = mean_abs(shap_values)

        summary_rel = None
        importance_rel = None
        if write_plots:
            importance_path = dest / "importance.png"
            plot_mean_abs_bar(
                feature_names,
                abs_mean,
                importance_path,
                title=f"{name}: mean |SHAP| ({used})",
            )
            importance_rel = str(importance_path.as_posix())
            beeswarm = dest / "summary.png"
            plotted = plot_shap_summary(
                shap_values,
                values_unscaled,
                feature_names,
                beeswarm,
                title=f"{name} SHAP summary ({label})",
            )
            if plotted is not None:
                summary_rel = str(plotted.as_posix())

        g_payload = global_payload(
            model_name=name,
            explainer=used,
            feature_names=feature_names,
            mean_abs_shap=abs_mean,
            n_samples=int(shap_values.shape[0]),
            data_origin=origin,
            occupancy_source=occ,
            data_label=label,
            explainer_mode=str(result["explainer_mode"]),
            expected_value=float(np.nanmean(base_values)) if np.isfinite(base_values).any() else None,
            summary_plot=summary_rel,
            rationale=str(result["rationale"]),
        )
        if importance_rel:
            g_payload["importance_plot"] = importance_rel
        if result.get("permutation_importance") is not None:
            perm = np.asarray(result["permutation_importance"], dtype=np.float64)
            g_payload["permutation_importance"] = [
                {"name": feat, "rmse_increase": float(val)}
                for feat, val in sorted(zip(feature_names, perm, strict=True), key=lambda kv: -kv[1])
            ]

        locals_out: list[dict[str, Any]] = []
        for i in range(len(preds)):
            ts_i = _iso_ts(ts_local[i]) if len(ts_local) else None
            b_id = str(groups_local[i]) if len(groups_local) else None
            if b_id == "_all":
                b_id = _meta_column(test_frame, "building_id", int(row_indices[i])) if int(row_indices[i]) < len(test_frame) else None
            origin_i = origin
            occ_i = occ
            idx_row = int(row_indices[i])
            if idx_row < len(test_frame):
                origin_i = _meta_column(test_frame, "data_origin", idx_row) or origin
                occ_i = _meta_column(test_frame, "occupancy_source", idx_row) or occ
                if b_id is None:
                    b_id = _meta_column(test_frame, "building_id", idx_row)
                if ts_i is None:
                    ts_i = _iso_ts(test_frame.iloc[idx_row]["ts"]) if "ts" in test_frame.columns else None
            expected = float(base_values[i]) if i < len(base_values) and np.isfinite(base_values[i]) else None
            locals_out.append(
                local_payload(
                    prediction=float(preds[i]),
                    feature_names=feature_names,
                    feature_values=values_unscaled[i],
                    contributions=shap_values[i],
                    model_name=name,
                    explainer=used,
                    building_id=b_id,
                    ts=ts_i,
                    data_origin=origin_i,
                    occupancy_source=occ_i,
                    expected_value=expected,
                    row_index=idx_row,
                    data_label=label,
                    explainer_mode=str(result["explainer_mode"]),
                )
            )

        write_model_explanations(artifacts_dir, name, global_payload=g_payload, local_payloads=locals_out)
        if persist_db:
            try_insert_explanation(method=used, payload={"model_name": name, "global": g_payload, "n_local": len(locals_out)})
        models_out.append(
            {
                "model_name": name,
                "explainer": used,
                "n_local": len(locals_out),
                "n_shap_rows": int(shap_values.shape[0]),
            }
        )
        LOGGER.info("%s wrote %s local explanations (%s)", name, len(locals_out), used)

    by_name = {row["model_name"]: row for row in models_out}
    explanations_root = Path(artifacts_dir) / "explanations"
    if explanations_root.exists():
        for child in explanations_root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if name in by_name:
                continue
            gpath = child / "global.json"
            lpath = child / "local.json"
            if not (gpath.exists() and lpath.exists()):
                continue
            try:
                gpayload = json.loads(gpath.read_text(encoding="utf-8"))
                lpayload = json.loads(lpath.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            by_name[name] = {
                "model_name": name,
                "explainer": gpayload.get("explainer"),
                "n_local": len(lpayload) if isinstance(lpayload, list) else 0,
                "n_shap_rows": gpayload.get("n_samples"),
            }

    manifest = {
        "data_source": data_source,
        "data_label": label,
        "random_seed": int(random_seed),
        "feature_names": feature_names,
        "occupancy_source": occ,
        "data_origin": origin,
        "models": list(by_name.values()),
        "explainer_policy": json_safe(EXPLAINER_SPECS),
    }
    write_manifest(artifacts_dir, manifest)
    return manifest
