const numberFmt = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 2 });
const kwhFmt = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const pctFmt = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 1, signDisplay: "exceptZero" });
const metricFmt = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 3 });

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-GB", { maximumFractionDigits: digits }).format(value);
}

export function formatKwh(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${kwhFmt.format(value)} kWh`;
}

export function formatTempC(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${numberFmt.format(value)} °C`;
}

export function formatHumidity(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${numberFmt.format(value)}%`;
}

export function formatOccupancy(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  // Historical features store a 0–1 profile; MQTT simulator publishes 0–100 percent.
  const pct = value <= 1 ? value * 100 : value;
  return `${numberFmt.format(pct)}% of typical capacity`;
}

export function formatPct(value: number | null | undefined, alreadyPercent = true): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const pct = alreadyPercent ? value : value * 100;
  return `${pctFmt.format(pct)}%`;
}

export function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return metricFmt.format(value);
}

export function parseTs(value: string | number | Date | null | undefined): Date | null {
  if (value === null || value === undefined) return null;
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatTs(value: string | number | Date | null | undefined): string {
  const date = parseTs(value);
  if (!date) return "—";
  return new Intl.DateTimeFormat("en-GB", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
    timeZone: "UTC",
  }).format(date) + " UTC";
}

export function formatTsShort(value: string | number | Date | null | undefined): string {
  const date = parseTs(value);
  if (!date) return "—";
  return new Intl.DateTimeFormat("en-GB", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
    timeZone: "UTC",
  }).format(date);
}

export function formatSignedKwh(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${kwhFmt.format(value)} kWh`;
}

export function isoFrom(date: Date): string {
  return date.toISOString();
}
