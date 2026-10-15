# Phase 4 — Explainable AI (SHAP)

Offline explanations only. The API **loads saved JSON**; it does not train and it does **not** run KernelSHAP on every HTTP request. Numbers in `artifacts/<run>/explanations/` come from the `shap` library (or, for linear models, the equivalent coefficient formula). This document does not contain example SHAP values from a canned demo.

## What SHAP is

SHAP (SHapley Additive exPlanations) attributes a model's prediction to input features using Shapley values from cooperative game theory. For a prediction \(f(x)\) and baseline \(E[f]\),

\[
f(x) \approx E[f] + \sum_i \phi_i(x)
\]

where \(\phi_i\) is feature \(i\)'s contribution on that row. A **positive** \(\phi_i\) is associated with a higher predicted next-hour kWh; a **negative** \(\phi_i\) is associated with a lower predicted kWh. That is an accounting identity for the fitted model, **not** a causal claim that the building used more energy *because* of that variable.

## Why SHAP is used here

PowerPredict is an academic energy-forecasting project. Operators and markers need to see **which engineered features** (lags, weather, calendar, simulated occupancy) the model used for a given \(y_{t+1}\) forecast. TreeSHAP and linear SHAP are exact for their model classes and fast enough to precompute. Deep sequence models are explained **offline** on a test subset so the API stays a file loader.

## Global vs local

| Kind | Meaning | Artifact |
| --- | --- | --- |
| **Global** | Mean absolute SHAP over the explained rows: \(\mathrm{mean}_n |\phi_i|\). Ranking of which features the model used most on that subset. Summary / importance plots under `explanations/<model>/`. | `explanations/<model>/global.json` |
| **Local** | One prediction: predicted kWh, each feature's value, signed contribution, positive vs negative lists, and a sentence built only from those contributions. | `explanations/<model>/local.json` |

Select a local row by `building_id` + `ts` or `row_index`. Missing artifacts → API **404**. No dummy arrays.

## Explainer choice per model

| Model | Explainer | Mode | Why |
| --- | --- | --- | --- |
| `linear_regression` | `shap.LinearExplainer` (fallback: \(\phi_i = \mathrm{coef}_i (x_i - E[x_i])\) from the fitted model) | Live-capable (precomputed in CLI) | Closed-form interventional SHAP; milliseconds. |
| `random_forest` | `shap.TreeExplainer` (TreeSHAP) | Live-capable (precomputed) | Polynomial-time exact Shapley values for trees. |
| `xgboost` | `shap.TreeExplainer` (TreeSHAP) | Live-capable (precomputed) | Same algorithm on the XGBoost ensemble. |
| `lstm` | `shap.PermutationExplainer` on flattened windows, then **sum over time** to one contribution per feature. Fallback: channel ablation vs background mean. | **Offline** | KernelSHAP-per-request is too slow and would not make LSTM “more explainable” at serving time. |
| `cnn_lstm` | Same as LSTM | **Offline** | CNN-LSTM is **not** claimed to be more explainable via live KernelSHAP. |

The HTTP API never instantiates KernelExplainer. Sequence explanations are stored after `python -m ml.explain`.

SHAP is computed in **scaled feature space** (the `StandardScaler` fit on train only). Payload `value` fields are the **unscaled** Phase 2 column (last timestep of the window for sequence models). Contributions are in **kWh** (model output units, after any target inverse-scale for LSTM/CNN-LSTM).

## Occupancy

BDG2 has no measured occupancy. The `occupancy` column is a schedule profile with `occupancy_source=simulated`. If that feature ranks high, the narrative says **simulated occupancy profile**, never “measured occupancy” or “occupancy sensors caused …”.

## Natural language

`ml.explain.narrative.narrative_from_contributions` uses **only** the contribution list it is given (top-\(k\) by \(|\phi|\)). Allowed language: “primarily influenced by”, “contributed +/− to this prediction”, “calculated associations … not proven causes”. Tests may pass a fake list to check the template; that is not a fake SHAP run.

## Limitations

- **Correlation, not causation.** \(\phi_{\text{temperature}} > 0\) means the model associated that temperature (relative to the background) with a higher forecast, not that temperature caused consumption.
- **TreeSHAP vs deep learning.** TreeSHAP is exact for RF/XGBoost. LSTM/CNN-LSTM explanations are permutation SHAP (or ablation) on a **subset**, aggregated across the sequence. They are coarser and slower; that is why they are offline.
- **Background / subset dependence.** Linear and permutation SHAP depend on the train background sample. Global rankings are for the explained test rows, not “importance on all of BDG2” unless that subset was used.
- **Occupancy is simulated** unless a future measured source exists.
- **KernelSHAP cost.** Model-agnostic KernelSHAP needs hundreds to thousands of model evaluations per row. It is not used on the request path. Flattened permutation SHAP for a window of length \(T\) and \(F\) features needs at least \(2TF+1\) evaluations per row; the CLI raises `sequence_max_evals` to that minimum so `PermutationExplainer` can run. If it still fails, channel ablation vs the train background mean is stored and labeled `permutation_ablation` (computed, not hardcoded).
- **TEST FIXTURE vs BDG2.** If the run’s `data_label` is `TEST FIXTURE`, do not present the numbers as BDG2 campus results. Fixture smoke models are labeled as such.

## How to run

Trained artifacts must already exist (`python -m ml.train`). Then, from the repo root:

```bash
pip install -r ml/requirements.txt
python -m ml.explain --artifacts-dir artifacts --processed-dir data/processed
# or, without rewriting data/processed, against an in-memory TEST FIXTURE:
python -m ml.explain --fixture --artifacts-dir artifacts/_phase4_fixture_shap --no-db
```

Useful flags: `--models linear_regression,random_forest,xgboost`, `--local-max 8`, `--background-max 64`, `--data-label "TEST FIXTURE"`, `--no-plots`, `--no-db`.

Writes:

```
artifacts/<run>/explanations/
  manifest.json
  <model>/global.json
  <model>/local.json
  <model>/importance.png
  <model>/summary.png
```

Images are gitignored with the rest of `artifacts/`.

Point the API at that run with `ARTIFACTS_DIR` (default: repo `artifacts/`).

```
GET /api/v1/models/{name}/explanations/global
GET /api/v1/models/{name}/explanations?building_id=&ts=
GET /api/v1/models/{name}/explanations?row_index=0
```

JWT required (same as Phase 1). CORS and rate limits unchanged. 404 if the JSON file is missing or no row matches.

## Tests

```bash
pytest tests/ml tests/backend/test_explanations.py
```

CI uses tiny trained fixture models. Full BDG2 is not required. Narrative tests use a provided contribution list (template only).
