# Artifacts

Trained models, `metrics.json`, and evaluation plots land here **after offline training**. The API will load files from this directory; it will not train.

```
artifacts/
  metrics.json
  feature_scaler.joblib
  linear_regression/model.joblib
  random_forest/model.joblib
  xgboost/model.joblib
  lstm/model.pt
  cnn_lstm/model.pt
  plots/
  explanations/
    manifest.json
    <model>/global.json
    <model>/local.json
    <model>/summary.png
    <model>/importance.png
```

Do not commit model binaries or images. Do not write dummy MAE/RMSE/R² or dummy SHAP arrays into this folder — empty until a real training / `python -m ml.explain` run produces them. See [docs/PHASE3_MODELS.md](../docs/PHASE3_MODELS.md) and [docs/PHASE4_SHAP.md](../docs/PHASE4_SHAP.md).
