import type { DataOrigin } from "../origin";

export type UserRole = "viewer" | "admin";

export type Building = {
  id: string;
  site_id: string;
  primary_use: string | null;
  sqm: number | null;
  timezone: string;
  selected_reason: string | null;
};

export type Reading = {
  id: string;
  building_id: string;
  ts: string;
  energy_kwh: number;
  data_origin: DataOrigin;
  meter_type: string;
  temperature_c: number | null;
  humidity_pct: number | null;
  occupancy: number | null;
  occupancy_source: string | null;
  schema_version: string | null;
};

export type ModelRun = {
  id: string;
  model_name: string;
  created_at: string;
  train_start: string | null;
  train_end: string | null;
  val_start: string | null;
  val_end: string | null;
  test_start: string | null;
  test_end: string | null;
  mae: number | null;
  rmse: number | null;
  r2: number | null;
  mape: number | null;
  artifact_path: string | null;
  random_seed: number | null;
  training_time_sec: number | null;
};

export type ModelPerformance = {
  model_name: string;
  mae: number | null;
  rmse: number | null;
  r2: number | null;
  mape: number | null;
  id: string | null;
  created_at: string | null;
  train_start: string | null;
  train_end: string | null;
  val_start: string | null;
  val_end: string | null;
  test_start: string | null;
  test_end: string | null;
  artifact_path: string | null;
  random_seed: number | null;
  training_time_sec: number | null;
  n: number | null;
  source: string;
  data_source: string | null;
  smoke: boolean | null;
};

export type Forecast = {
  id: string;
  building_id: string;
  model_run_id: string | null;
  target_ts: string;
  y_true: number | null;
  y_pred: number;
  split: string;
  model_name: string | null;
};

export type Anomaly = {
  id: string;
  building_id: string;
  ts: string;
  score: number;
  is_anomaly: boolean;
  model_run_id: string | null;
  data_origin: DataOrigin;
  deviation_pct: number | null;
  severity: string | null;
  actual_kwh: number | null;
  expected_kwh: number | null;
};

export type IotStatus = {
  status: "LIVE" | "SIMULATED" | "OFFLINE";
  broker_connected: boolean;
  last_message_at: string | null;
  last_data_origin: DataOrigin | null;
  last_building_id: string | null;
  topic: string;
  mqtt_enabled: boolean;
  note: string;
};

export type FeatureRank = {
  name: string;
  mean_abs_shap: number;
};

export type FeatureContribution = {
  name: string;
  value: number;
  contribution: number;
  shap_value?: number;
};

export type LiveLocalExplanation = {
  prediction: number;
  prediction_kwh: number;
  base_value: number;
  features: FeatureContribution[];
  model: string;
  model_name: string;
  building_id: string;
  explainer: string;
  explainer_mode: string | null;
  occupancy_source: string | null;
  narrative: string | null;
  live: boolean;
  warning: string | null;
};

export type GlobalExplanation = {
  model_name: string;
  explainer: string;
  explainer_mode: string | null;
  rationale: string | null;
  n_samples: number;
  ranking: FeatureRank[];
  expected_value: number | null;
  data_origin: string | null;
  occupancy_source: string | null;
  data_label: string | null;
  summary_plot: string | null;
  importance_plot: string | null;
  narrative: string | null;
};

export type LocalExplanation = {
  prediction: number;
  features: FeatureContribution[];
  model_name: string;
  building_id: string | null;
  ts: string | null;
  explainer: string;
  explainer_mode: string | null;
  data_origin: string | null;
  occupancy_source: string | null;
  expected_value: number | null;
  row_index: number | null;
  data_label: string | null;
  narrative: string | null;
  positive_contributions?: FeatureContribution[];
  negative_contributions?: FeatureContribution[];
};

export type Health = {
  status: "ok";
};
