export const MODEL_ORDER = [
  "linear_regression",
  "random_forest",
  "xgboost",
  "lstm",
  "cnn_lstm",
] as const;

export type ModelName = (typeof MODEL_ORDER)[number];

export const MODEL_LABELS: Record<string, string> = {
  linear_regression: "Linear regression",
  random_forest: "Random forest",
  xgboost: "XGBoost",
  lstm: "LSTM",
  cnn_lstm: "CNN-LSTM",
};

export function isModelName(value: string): value is ModelName {
  return (MODEL_ORDER as readonly string[]).includes(value);
}

export function modelLabel(name: string | null | undefined): string {
  if (!name) return "Unknown model";
  return MODEL_LABELS[name] ?? name;
}
