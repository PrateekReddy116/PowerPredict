# XAI / SHAP audit (before code changes)

Audit of What-if → predict and Explain → stored SHAP. No SHAP values were fabricated for this document.

## Verdict: Explain page is (D) a mix

| Piece | What the UI actually shows |
| --- | --- |
| Global chart / ranking | **(A)** stored **mean \|SHAP\|** from `artifacts/explanations/<model>/global.json` |
| Local table / local narrative | **(C)** one **cached stored** row from `local.json` (requested as `row=0`) |
| Live local for the current What-if vector | **Not used** |

The user’s suspicion is confirmed: **stored global + one stored local example**. It is not live local SHAP for the What-if form.

## Architecture

```
What-if (Predict.tsx)
  POST /api/v1/predict  (or /predict/horizon)
    → app.services.predict.predict_single / predict_horizon
    → feature_scaler.joblib feature_names + StandardScaler
    → _build_feature_vector (user weather/occupancy/energy + defaults)
    → model.predict(scaled row)  [sequence: tiled seq_len copies]
  No SHAP call.

Explain (Explain.tsx)
  GET /api/explainability?model=…
    → load_global → explanations/<model>/global.json
  GET /api/explainability?model=…&row=0
    → load_locals + find_local(row_index=0) → explanations/<model>/local.json
  Numbers come from the offline CLI (`python -m ml.explain`). The API does not recompute SHAP on GET.
```

Routers: `backend/app/api/explainability.py` is mounted at both `/api` and `/api/v1` (same GET). Duplicate GET also exists at `/api/v1/models/{name}/explanations/global` and `.../explanations`. Frontend Explain uses `/api/explainability`.

## Does What-if currently update XAI?

**No.** `Predict.tsx` stores `singleResult` / `horizonResult` only. Changing inputs and clicking Predict refreshes the kWh number (and the horizon chart). It never fetches SHAP, never draws SHAP bars, never updates NL text from Shapley values.

## Caching / staleness

| Layer | Behaviour |
| --- | --- |
| GET explainability | File load only. Never KernelSHAP. Never dummy arrays (404 if missing). |
| Stored JSON | Written by `ml.explain.run`; stays until the CLI is re-run. |
| Frontend Explain | Refetches when `model` / retry nonce changes. Does not poll. Independent of What-if state. |
| Frontend What-if | No explanation state at all. |
| Hardcoded SHAP | No canned SHAP arrays in the API. NL templates exist (`ml.explain.narrative`) and are filled from computed or stored lists. |
| Global narrative bug | `global_payload` feeds **mean \|SHAP\|** into `narrative_from_contributions`, which says “contributed +X to **this prediction**”. That is the wrong wording for a global average of absolute values (always ≥ 0). |

## Feature name / order vs training

- Training matrix order is `ml.pipeline.FEATURE_COLUMNS` (energy at *t*, lags, past-only rolling, temporal, weather, occupancy, sqm). Target `target_energy_kwh` is `shift(-1)` and is **not** in `FEATURE_COLUMNS`.
- Saved scaler bundle: `feature_scaler.joblib` → `{scaler, feature_names, seq_len}`.
- Live predict builds `x` as `[feature_map[col] for col in scaler feature_names]` then `scaler.transform`. Runtime order follows the **scaler**, not a hardcoded list. `_DEFAULT_FEATURE_COLUMNS` in `predict.py` is documentation-only (unused if the scaler exists).
- Offline SHAP (`ml.explain.run.prepare_for_explain`) reorders unscaled X to the same scaler `feature_names` before scaling.

**Alignment risk (inference vs training construction), not a y_{t+1} leak:**

- Training lags are true `energy[t-k]`. Training rolling is **past-only** (`shift(1)` then `rolling`; `ROLLING_INCLUDES_CURRENT_T = False`).
- What-if fills **every** `lag_*` and `rolling_*` with the **same** user `energy_kwh` (or 0). That is a constant-history approximation, not the training lag/rolling recipe.
- Sequence models tile one scaled row `seq_len` times (“no real history”).

`energy_kwh` as a **current-hour predictor** is the intended lag-0 feature at time *t*. The target is energy at *t+1*. That is **not** leaking `y_{t+1}` into X. Predict never includes `target_energy_kwh`.

## Explainer map (CLI vs HTTP)

| Model | Registered explainer | CLI | HTTP GET (today) | HTTP live local (today) |
| --- | --- | --- | --- | --- |
| linear_regression | LinearExplainer (fallback: φ = coef × (x − E[x])) | yes | stored JSON | **none** |
| random_forest | TreeExplainer | yes | stored JSON | **none** |
| xgboost | TreeExplainer | yes | stored JSON | **none** |
| lstm / cnn_lstm | PermutationExplainer offline (flatten window, sum over time; ablation fallback). **Never KernelSHAP** | yes | stored JSON | **none** |

SHAP in the CLI is computed in **scaled** feature space; payload `value` fields are **unscaled**. Contributions are in kWh (model output units).

**LSTM live SHAP:** Flattened permutation needs ≥ `2 * seq_len * n_features + 1` model evals per row (e.g. 24 × ~28 features → ~1300+ forwards). Too expensive for an interactive HTTP request. KernelSHAP is forbidden on the request path. **Limitation: live local SHAP for LSTM/CNN-LSTM should be 501 with a clear message**, not KernelSHAP and not a copy of `local.json`.

Occupancy is simulated (`occupancy_source`); UI/API must keep that label.

## Files traced

- `frontend/src/pages/Predict.tsx`, `frontend/src/pages/Explain.tsx`
- `backend/app/api/v1/predict.py`, `backend/app/api/explainability.py`, `backend/app/api/v1/models.py`
- `backend/app/services/predict.py`, `backend/app/services/explanations.py`
- `ml/explain/compute.py`, `registry.py`, `run.py`, `payload.py`, `narrative.py`
- `ml/pipeline.py` (`FEATURE_COLUMNS`, `add_target`), `ml/features/rolling.py`

## Intended Phase 2 fixes (implementation follows this file)

1. `POST /api/v1/explainability/local` (same router as GET; also available under `/api/explainability/local`). JWT. Same what-if body as predict. Shared feature-vector helper with `predict.py`. Tree/Linear explainer on **that one row**. Never return stored `local.json`. 404 if no model/scaler artifact. 501 for LSTM/CNN-LSTM.
2. Keep GET global as stored mean |SHAP|; label it global.
3. What-if: after Predict, call live local explain on the same payload; show prediction + “Why this prediction?” from that response.
4. Explain page: global = mean |SHAP|; stored local captioned as CLI example, not live What-if. NL: global “highest average influence”; local “contributed approximately X kWh to this prediction”.
5. Pytest: two what-if scenarios, different pred and shap; feature order = scaler names; no target column; wording tests.
