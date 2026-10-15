"""Offline explanation CLI: `python -m ml.explain`.

Computes global + sample local SHAP into artifacts/<run>/explanations/.
Does not invent SHAP values. Does not run KernelSHAP per HTTP request.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ml.config import load_config
from ml.models import ALL_MODEL_NAMES
from ml.paths import DEFAULT_ARTIFACTS_DIR, DEFAULT_CONFIG, REPO_ROOT

LOGGER = logging.getLogger("ml.explain")


def _resolve(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute SHAP explanations for saved PowerPredict models")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", default=None, help="comma-separated model names (default: whatever artifacts exist)")
    parser.add_argument("--data", type=Path, default=None, help="parquet/csv feature matrix")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--building-id", default=None)
    parser.add_argument("--fixture", action="store_true", help="Build in-memory TEST FIXTURE features (does not rewrite data/processed)")
    parser.add_argument("--force-ratio-split", action="store_true")
    parser.add_argument("--background-max", type=int, default=None)
    parser.add_argument("--local-max", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--data-label", default=None, help="e.g. 'TEST FIXTURE' or 'BDG2'")
    return parser


def _fixture_splits():
    from ml.ingest.fixture import (
        make_fixture_electricity_wide,
        make_fixture_metadata,
        make_fixture_weather,
    )
    from ml.pipeline import run_pipeline

    result = run_pipeline(
        make_fixture_metadata(),
        make_fixture_electricity_wide(),
        make_fixture_weather(),
        source="test_fixture",
        n_buildings=2,
    )
    return result.train, result.val, result.test, "test_fixture:in-memory", "TEST FIXTURE"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from ml.ingest.db import load_env_file

    load_env_file()
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    paths = config.get("paths") or {}
    explain_cfg = config.get("explain") or {}
    artifacts_dir = _resolve(args.artifacts_dir or paths.get("artifacts_dir") or DEFAULT_ARTIFACTS_DIR)
    names = None
    if args.models:
        names = [n.strip() for n in str(args.models).split(",") if n.strip()]
        unknown = [n for n in names if n not in ALL_MODEL_NAMES]
        if unknown:
            LOGGER.error("unknown models %s", unknown)
            return 2

    background_max = int(args.background_max or explain_cfg.get("background_max") or 64)
    local_max = int(args.local_max or explain_cfg.get("local_max") or 8)
    seq_evals = explain_cfg.get("sequence_max_evals")
    seed = int(config.get("random_seed", 42))

    from ml.explain.run import run_explanations

    try:
        if args.fixture:
            train, val, test, source, label = _fixture_splits()
            run_explanations(
                artifacts_dir=artifacts_dir,
                train=train,
                val=val,
                test=test,
                config=config,
                model_names=names,
                data_source=source,
                data_label=args.data_label or label,
                background_max=background_max,
                local_max=local_max,
                sequence_max_evals=int(seq_evals) if seq_evals is not None else None,
                random_seed=seed,
                write_plots=not args.no_plots,
                persist_db=not args.no_db,
            )
        elif args.data is not None:
            from ml.models.dataset import load_feature_matrix

            data_path = _resolve(args.data)
            frame = load_feature_matrix(data_path)
            run_explanations(
                artifacts_dir=artifacts_dir,
                frame=frame,
                config=config,
                model_names=names,
                data_source=data_path.as_posix(),
                data_label=args.data_label,
                background_max=background_max,
                local_max=local_max,
                sequence_max_evals=int(seq_evals) if seq_evals is not None else None,
                random_seed=seed,
                write_plots=not args.no_plots,
                persist_db=not args.no_db,
                force_ratio=bool(args.force_ratio_split),
            )
        else:
            from ml.models.dataset import load_processed_frame, load_processed_splits

            processed = _resolve(args.processed_dir or paths.get("processed_dir") or "data/processed")
            splits = load_processed_splits(processed, building_id=args.building_id)
            label = args.data_label
            source = processed.as_posix()
            summary = processed / "pipeline_summary.json"
            if summary.exists():
                import json

                try:
                    payload = json.loads(summary.read_text(encoding="utf-8"))
                    src = payload.get("source")
                    if src:
                        source = f"{src}:{processed.as_posix()}"
                    if label is None and str(src).lower() in {"test_fixture", "fixture"}:
                        label = "TEST FIXTURE"
                    if label is None and str(src).lower() == "bdg2":
                        label = "BDG2"
                except (OSError, json.JSONDecodeError):
                    pass
            if splits is not None:
                run_explanations(
                    artifacts_dir=artifacts_dir,
                    train=splits[0],
                    val=splits[1],
                    test=splits[2],
                    config=config,
                    model_names=names,
                    data_source=source,
                    data_label=label,
                    background_max=background_max,
                    local_max=local_max,
                    sequence_max_evals=int(seq_evals) if seq_evals is not None else None,
                    random_seed=seed,
                    write_plots=not args.no_plots,
                    persist_db=not args.no_db,
                )
            else:
                frame = load_processed_frame(processed, building_id=args.building_id)
                run_explanations(
                    artifacts_dir=artifacts_dir,
                    frame=frame,
                    config=config,
                    model_names=names,
                    data_source=source,
                    data_label=label,
                    background_max=background_max,
                    local_max=local_max,
                    sequence_max_evals=int(seq_evals) if seq_evals is not None else None,
                    random_seed=seed,
                    write_plots=not args.no_plots,
                    persist_db=not args.no_db,
                    force_ratio=bool(args.force_ratio_split),
                )
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        LOGGER.error("Need trained artifacts and a feature matrix. Do not invent SHAP values.")
        return 2
    LOGGER.info("wrote %s", artifacts_dir / "explanations" / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
