import type { DataOrigin } from "../origin";

export type Grain = "hourly" | "daily" | "weekly" | "monthly" | "weekday";

export const GRAIN_LABELS: Record<Grain, string> = {
  hourly: "Hourly (as returned)",
  daily: "Daily totals — aggregated from hourly history",
  weekly: "ISO-week totals — aggregated from hourly history",
  monthly: "Monthly totals — aggregated from hourly history",
  weekday: "Mean by weekday — aggregated from hourly history",
};

export type EnergyPoint = {
  ts: string;
  energy_kwh: number;
  data_origin: DataOrigin;
  temperature_c?: number | null;
};

export type AggregatedPoint = {
  key: string;
  label: string;
  kwh: number;
  n: number;
  mean_kwh: number;
  origin: DataOrigin | "mixed";
  temperature_c: number | null;
};

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function utcDate(ts: string): Date {
  return new Date(ts);
}

function pad(n: number): string {
  return String(n).padStart(2, "0");
}

function isoWeekKey(date: Date): string {
  const utc = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
  const day = utc.getUTCDay() || 7;
  utc.setUTCDate(utc.getUTCDate() + 4 - day);
  const yearStart = new Date(Date.UTC(utc.getUTCFullYear(), 0, 1));
  const week = Math.ceil(((utc.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
  return `${utc.getUTCFullYear()}-W${pad(week)}`;
}

function bucketKey(date: Date, grain: Grain): string {
  const y = date.getUTCFullYear();
  const m = pad(date.getUTCMonth() + 1);
  const d = pad(date.getUTCDate());
  const h = pad(date.getUTCHours());
  if (grain === "hourly") return `${y}-${m}-${d} ${h}:00`;
  if (grain === "daily") return `${y}-${m}-${d}`;
  if (grain === "weekly") return isoWeekKey(date);
  if (grain === "monthly") return `${y}-${m}`;
  return WEEKDAYS[date.getUTCDay()] ?? "Sun";
}

function mergeOrigin(current: DataOrigin | "mixed" | undefined, next: DataOrigin): DataOrigin | "mixed" {
  if (!current) return next;
  if (current === "mixed" || current !== next) return "mixed";
  return current;
}

export function aggregateEnergy(points: EnergyPoint[], grain: Grain): AggregatedPoint[] {
  if (grain === "hourly") {
    return points.map((point) => ({
      key: point.ts,
      label: point.ts,
      kwh: point.energy_kwh,
      n: 1,
      mean_kwh: point.energy_kwh,
      origin: point.data_origin,
      temperature_c: point.temperature_c ?? null,
    }));
  }

  type Acc = {
    kwh: number;
    n: number;
    origin: DataOrigin | "mixed";
    tempSum: number;
    tempN: number;
  };
  const buckets = new Map<string, Acc>();
  for (const point of points) {
    const date = utcDate(point.ts);
    if (Number.isNaN(date.getTime())) continue;
    const key = bucketKey(date, grain);
    const prev = buckets.get(key);
    const temp = point.temperature_c;
    const hasTemp = temp !== null && temp !== undefined && !Number.isNaN(temp);
    if (!prev) {
      buckets.set(key, {
        kwh: point.energy_kwh,
        n: 1,
        origin: point.data_origin,
        tempSum: hasTemp ? temp : 0,
        tempN: hasTemp ? 1 : 0,
      });
    } else {
      prev.kwh += point.energy_kwh;
      prev.n += 1;
      prev.origin = mergeOrigin(prev.origin, point.data_origin);
      if (hasTemp) {
        prev.tempSum += temp;
        prev.tempN += 1;
      }
    }
  }

  const keys = [...buckets.keys()];
  if (grain === "weekday") {
    keys.sort((a, b) => WEEKDAYS.indexOf(a) - WEEKDAYS.indexOf(b));
  } else {
    keys.sort();
  }

  return keys.map((key) => {
    const acc = buckets.get(key)!;
    return {
      key,
      label: key,
      kwh: grain === "weekday" ? acc.kwh / acc.n : acc.kwh,
      n: acc.n,
      mean_kwh: acc.kwh / acc.n,
      origin: acc.origin,
      temperature_c: acc.tempN ? acc.tempSum / acc.tempN : null,
    };
  });
}

export type TempEnergyPoint = {
  temperature_c: number;
  energy_kwh: number;
  ts: string;
  data_origin: DataOrigin;
};

export function temperatureVsEnergy(points: EnergyPoint[]): TempEnergyPoint[] {
  const out: TempEnergyPoint[] = [];
  for (const point of points) {
    const temp = point.temperature_c;
    if (temp === null || temp === undefined || Number.isNaN(temp)) continue;
    out.push({
      temperature_c: temp,
      energy_kwh: point.energy_kwh,
      ts: point.ts,
      data_origin: point.data_origin,
    });
  }
  return out;
}

export function downsample<T>(items: T[], maxPoints: number): { items: T[]; factor: number } {
  if (items.length <= maxPoints) return { items, factor: 1 };
  const factor = Math.ceil(items.length / maxPoints);
  return { items: items.filter((_, index) => index % factor === 0), factor };
}
