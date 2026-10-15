export type DataOrigin = "historical" | "simulated_iot" | "real_iot";

export const ORIGIN_LABELS: Record<DataOrigin, string> = {
  historical: "BDG2 archive",
  simulated_iot: "MQTT simulator",
  real_iot: "Live meter",
};

export const ORIGIN_HINTS: Record<DataOrigin, string> = {
  historical: "Cleaned electricity from Building Data Genome 2",
  simulated_iot: "Synthetic stream published to Mosquitto",
  real_iot: "Physical meter bridge — not a simulator",
};

export function isDataOrigin(value: string | null | undefined): value is DataOrigin {
  return value === "historical" || value === "simulated_iot" || value === "real_iot";
}

export function parseOrigin(value: string | null | undefined): DataOrigin | null {
  return isDataOrigin(value) ? value : null;
}
