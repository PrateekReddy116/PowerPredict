import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { apiPostAuth, apiPostAuthOptional, ApiError, AuthError } from "../api/client";
import type { LiveLocalExplanation } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { ChartPanel } from "../components/ChartPanel";
import { DeskTooltip } from "../components/DeskTooltip";
import { ErrorBanner } from "../components/ErrorBanner";
import { OccupancyCaption } from "../components/OccupancyCaption";
import { PageHeader } from "../components/PageHeader";
import { SelectField } from "../components/SelectField";
import { useBuildings } from "../hooks/useBuildings";
import { useSelectedBuilding } from "../hooks/useSelectedBuilding";
import { MODEL_SELECT_OPTIONS } from "../hooks/useSelectedModel";
import { SERIES } from "../lib/colors";
import { formatKwh, formatMetric } from "../lib/format";
import { MODEL_ORDER, modelLabel, type ModelName } from "../lib/models";
import { occupancyFeatureLabel } from "../lib/occupancy";

type PredictResponse = {
  prediction_kwh: number;
  model: string;
  building_id: string;
  horizon_hours: number;
  inputs_used: Record<string, number | boolean | null>;
  warning: string | null;
  defaults_note: string;
};

type HorizonPoint = {
  hour_ahead: number;
  prediction_kwh: number;
  timestamp_utc: string;
};

type HorizonResponse = {
  building_id: string;
  model: string;
  horizon_hours: number;
  method: string;
  note: string;
  warning: string;
  predictions: HorizonPoint[];
};

type CompareRow = {
  model: ModelName;
  status: "ok" | "missing" | "error";
  prediction_kwh?: number;
  message?: string;
};

const HORIZON_OPTIONS = [
  { value: "1", label: "1 hour" },
  { value: "6", label: "6 hours" },
  { value: "24", label: "24 hours" },
  { value: "48", label: "48 hours" },
];

/** Models queried for the 1h side-by-side panel (tabular + sequence if trained). */
const COMPARE_MODELS: ModelName[] = [...MODEL_ORDER];

function randBetween(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function round1(n: number): string {
  return (Math.round(n * 10) / 10).toString();
}

/**
 * Plausible random what-if inputs. Occupancy is always a simulated 0–100 score,
 * not a measured sensor reading.
 */
function randomWhatIfInputs() {
  return {
    temperature: round1(randBetween(8, 36)),
    humidity: String(Math.round(randBetween(30, 90))),
    occupancy: String(Math.round(randBetween(0, 100))),
    energyKwh: round1(randBetween(25, 280)),
    windSpeed: round1(randBetween(0, 12)),
    hour: String(Math.floor(randBetween(0, 24))),
    dayOfWeek: String(Math.floor(randBetween(0, 7))),
  };
}

function buildPredictBody(
  buildingId: string,
  model: string,
  fields: {
    temperature: string;
    humidity: string;
    occupancy: string;
    energyKwh: string;
    windSpeed: string;
    hour: string;
    dayOfWeek: string;
  },
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    building_id: buildingId,
    model,
  };
  if (fields.temperature) body.temperature = parseFloat(fields.temperature);
  if (fields.humidity) body.humidity = parseFloat(fields.humidity);
  if (fields.occupancy) body.occupancy = parseFloat(fields.occupancy);
  if (fields.energyKwh) body.energy_kwh = parseFloat(fields.energyKwh);
  if (fields.windSpeed) body.wind_speed = parseFloat(fields.windSpeed);
  if (fields.hour) body.hour = parseInt(fields.hour, 10);
  if (fields.dayOfWeek) body.day_of_week = parseInt(fields.dayOfWeek, 10);
  return body;
}

export function PredictPage() {
  const { token, logout } = useAuth();
  const { buildings } = useBuildings();
  const { buildingId, setBuildingId } = useSelectedBuilding(buildings.map((b) => b.id));

  const [model, setModel] = useState("xgboost");
  const [horizon, setHorizon] = useState("1");
  const [temperature, setTemperature] = useState("");
  const [humidity, setHumidity] = useState("");
  const [occupancy, setOccupancy] = useState("");
  const [energyKwh, setEnergyKwh] = useState("");
  const [windSpeed, setWindSpeed] = useState("");
  const [hour, setHour] = useState("");
  const [dayOfWeek, setDayOfWeek] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [singleResult, setSingleResult] = useState<PredictResponse | null>(null);
  const [horizonResult, setHorizonResult] = useState<HorizonResponse | null>(null);
  const [compareRows, setCompareRows] = useState<CompareRow[] | null>(null);
  const [localExp, setLocalExp] = useState<LiveLocalExplanation | null>(null);
  const [explainNote, setExplainNote] = useState<string | null>(null);

  const buildingOptions = buildings.map((b) => ({
    value: b.id,
    label: b.primary_use ? `${b.id} · ${b.primary_use}` : b.id,
  }));

  const fieldBag = {
    temperature,
    humidity,
    occupancy,
    energyKwh,
    windSpeed,
    hour,
    dayOfWeek,
  };

  function fillRandom() {
    const next = randomWhatIfInputs();
    setTemperature(next.temperature);
    setHumidity(next.humidity);
    setOccupancy(next.occupancy);
    setEnergyKwh(next.energyKwh);
    setWindSpeed(next.windSpeed);
    setHour(next.hour);
    setDayOfWeek(next.dayOfWeek);
  }

  async function fetchCompare(
    authToken: string,
    baseBody: Record<string, unknown>,
  ): Promise<CompareRow[]> {
    const results = await Promise.all(
      COMPARE_MODELS.map(async (name): Promise<CompareRow> => {
        try {
          const res = await apiPostAuthOptional<PredictResponse>(
            "/api/v1/predict",
            { ...baseBody, model: name },
            authToken,
          );
          if (!res) {
            return { model: name, status: "missing", message: "Not trained" };
          }
          return { model: name, status: "ok", prediction_kwh: res.prediction_kwh };
        } catch (err: unknown) {
          if (err instanceof AuthError) throw err;
          return {
            model: name,
            status: "error",
            message: err instanceof Error ? err.message : "Failed",
          };
        }
      }),
    );
    return results;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!token || !buildingId || !model) return;
    setLoading(true);
    setError(null);
    setSingleResult(null);
    setHorizonResult(null);
    setCompareRows(null);
    setLocalExp(null);
    setExplainNote(null);

    const body = buildPredictBody(buildingId, model, fieldBag);
    const compareBody = buildPredictBody(buildingId, "xgboost", fieldBag);

    try {
      const h = parseInt(horizon, 10);
      if (h <= 1) {
        const res = await apiPostAuth<PredictResponse>("/api/v1/predict", body, token);
        setSingleResult(res);
      } else {
        const hBody = { ...body, horizon_hours: h };
        const res = await apiPostAuth<HorizonResponse>("/api/v1/predict/horizon", hBody, token);
        setHorizonResult(res);
      }

      try {
        const rows = await fetchCompare(token, compareBody);
        setCompareRows(rows);
      } catch (compareErr: unknown) {
        if (compareErr instanceof AuthError) {
          logout();
          return;
        }
        setCompareRows(null);
      }

      try {
        const local = await apiPostAuth<LiveLocalExplanation>(
          "/api/v1/explainability/local",
          body,
          token,
        );
        setLocalExp(local);
      } catch (explainErr: unknown) {
        setLocalExp(null);
        if (explainErr instanceof AuthError) {
          logout();
        } else if (explainErr instanceof ApiError && explainErr.status === 501) {
          setExplainNote(explainErr.message);
        } else {
          setExplainNote(
            explainErr instanceof Error ? explainErr.message : "Could not explain this prediction",
          );
        }
      }
    } catch (err: unknown) {
      if (err instanceof AuthError) {
        logout();
      } else {
        setError(err instanceof Error ? err.message : "Prediction failed");
        try {
          const rows = await fetchCompare(token, compareBody);
          setCompareRows(rows);
        } catch {
          /* ignore secondary compare failure */
        }
      }
    } finally {
      setLoading(false);
    }
  }

  const chartData = horizonResult?.predictions.map((p) => ({
    x: `+${p.hour_ahead}h`,
    kWh: p.prediction_kwh,
  }));

  const shapBars = localExp
    ? [...localExp.features]
        .map((row) => ({
          name: occupancyFeatureLabel(row.name, localExp.occupancy_source),
          shap: row.shap_value ?? row.contribution,
        }))
        .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
        .slice(0, 12)
    : [];

  return (
    <section>
      <PageHeader
        kicker="What-if prediction"
        title="Live inference"
        description="Provide your own input values and get a next-hour (or multi-hour) prediction plus a live local SHAP explanation for that same vector. This is NOT a stored CLI example. Occupancy is simulated (0–100), not a measured sensor."
      />

      <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
        <form onSubmit={handleSubmit} className="grid gap-4 sm:grid-cols-2">
          <SelectField
            id="predict-building"
            label="Building"
            value={buildingId ?? ""}
            onChange={setBuildingId}
            options={buildingOptions}
          />
          <SelectField
            id="predict-model"
            label="Primary model"
            value={model}
            onChange={setModel}
            options={MODEL_SELECT_OPTIONS}
          />
          <SelectField
            id="predict-horizon"
            label="Horizon (primary)"
            value={horizon}
            onChange={setHorizon}
            options={HORIZON_OPTIONS}
          />
          <div className="flex items-end">
            <p className="pb-2 text-xs leading-5 text-mute">
              Right panel always compares <span className="text-paper">next-hour</span> across
              models with the same inputs.
            </p>
          </div>

          <InputField
            label="Temperature (°C)"
            value={temperature}
            onChange={setTemperature}
            placeholder="e.g. 31.2"
          />
          <InputField
            label="Humidity (%)"
            value={humidity}
            onChange={setHumidity}
            placeholder="e.g. 67"
          />
          <InputField
            label="Occupancy (0–100, simulated)"
            value={occupancy}
            onChange={setOccupancy}
            placeholder="e.g. 75"
          />
          <InputField
            label="Energy kWh (current hour)"
            value={energyKwh}
            onChange={setEnergyKwh}
            placeholder="e.g. 120"
          />
          <InputField
            label="Wind speed (m/s)"
            value={windSpeed}
            onChange={setWindSpeed}
            placeholder="e.g. 5"
          />
          <InputField label="Hour (0-23)" value={hour} onChange={setHour} placeholder="auto" />
          <InputField
            label="Day of week (0=Mon)"
            value={dayOfWeek}
            onChange={setDayOfWeek}
            placeholder="auto"
          />

          <div className="sm:col-span-2 flex flex-wrap items-end gap-2">
            <button
              type="button"
              onClick={fillRandom}
              className="rounded-md border border-hair px-4 py-2 text-sm font-medium text-paper hover:border-copper"
            >
              Random
            </button>
            <button
              type="submit"
              disabled={loading || !buildingId}
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
            >
              {loading ? "Predicting…" : "Predict"}
            </button>
            <p className="w-full text-xs text-mute">
              Random fills temperature, humidity,{" "}
              <span className="text-sim">simulated occupancy (0–100)</span>, energy, wind, hour,
              and day of week with plausible values.
            </p>
          </div>
        </form>

        <aside className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
          <h3 className="font-display text-sm font-semibold text-paper">All models · next hour</h3>
          <p className="mt-1 text-xs leading-5 text-mute">
            Same input vector; one POST /api/v1/predict per model. Missing artifacts show as not
            trained — values are never invented.
          </p>
          {!compareRows && !loading ? (
            <p className="mt-6 text-sm text-mute">Run Predict to compare models.</p>
          ) : null}
          {loading && !compareRows ? (
            <p className="mt-6 font-mono text-sm text-mute">Loading comparisons…</p>
          ) : null}
          {compareRows ? (
            <ul className="mt-4 space-y-2">
              {compareRows.map((row) => {
                const selected = row.model === model;
                return (
                  <li
                    key={row.model}
                    className={`flex items-center justify-between gap-3 border-b border-hair/60 py-2 text-sm ${
                      selected ? "text-copper" : "text-paper"
                    }`}
                  >
                    <span className="font-mono text-xs">
                      {modelLabel(row.model)}
                      {selected ? " · primary" : ""}
                    </span>
                    {row.status === "ok" ? (
                      <span className="tabular-nums font-medium">
                        {formatKwh(row.prediction_kwh ?? null)}
                      </span>
                    ) : (
                      <span className="font-mono text-xs text-mute">
                        {row.message ?? (row.status === "missing" ? "Not trained" : "Error")}
                      </span>
                    )}
                  </li>
                );
              })}
            </ul>
          ) : null}
        </aside>
      </div>

      {error && <ErrorBanner message={error} />}

      {singleResult && (
        <div className="mt-6 rounded-lg border border-gray-700 bg-gray-800/50 p-5">
          <p className="text-xs uppercase tracking-[0.14em] text-mute">Primary prediction</p>
          <p className="mt-1 text-3xl font-bold text-indigo-300">
            {singleResult.prediction_kwh} <span className="text-lg font-normal text-gray-400">kWh</span>
          </p>
          <p className="mt-1 text-sm text-gray-400">
            Model: {singleResult.model} · Building: {singleResult.building_id} · Horizon:{" "}
            {singleResult.horizon_hours}h
          </p>
          {singleResult.warning && (
            <p className="mt-2 text-xs text-yellow-400">⚠ {singleResult.warning}</p>
          )}
          <details className="mt-3 text-xs text-gray-500">
            <summary className="cursor-pointer">Inputs used & defaults</summary>
            <pre className="mt-1 whitespace-pre-wrap">
              {JSON.stringify(singleResult.inputs_used, null, 2)}
            </pre>
            <p className="mt-1">{singleResult.defaults_note}</p>
          </details>
        </div>
      )}

      {horizonResult && chartData && (
        <div className="mt-6">
          <p className="mb-2 text-xs uppercase tracking-[0.14em] text-mute">
            Primary model · {horizonResult.horizon_hours}h horizon
          </p>
          <ChartPanel
            height={300}
            empty={chartData.length === 0 ? "No predictions" : null}
            caption={`Recursive forecast — ${horizonResult.note}`}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 12, right: 16, left: 8, bottom: 8 }}>
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" vertical={false} />
                <XAxis
                  dataKey="x"
                  tick={{ fill: "#8aa0b0", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{ fill: "#8aa0b0", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={48}
                />
                <Tooltip content={<DeskTooltip valueSuffix="kWh" />} />
                <Line
                  type="monotone"
                  dataKey="kWh"
                  name="Predicted kWh"
                  stroke={SERIES.predicted}
                  dot={false}
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
          <p className="mt-2 text-xs text-yellow-400">⚠ {horizonResult.warning}</p>
        </div>
      )}

      {explainNote ? <ErrorBanner message={explainNote} /> : null}

      {localExp ? (
        <div className="mt-6 space-y-4">
          <h3 className="text-lg font-semibold text-gray-100">Why this prediction?</h3>
          <p className="text-sm text-gray-400">
            Live local SHAP for the <span className="text-paper">selected</span> model on this
            What-if vector ({localExp.explainer}). Signed bars are contributions to{" "}
            <span className="tabular-nums">{formatKwh(localExp.prediction)}</span>, not global mean
            |SHAP|.
          </p>
          <OccupancyCaption source={localExp.occupancy_source} />
          {localExp.warning ? <p className="text-xs text-yellow-400">⚠ {localExp.warning}</p> : null}
          {localExp.narrative ? (
            <p className="max-w-3xl text-sm leading-6 text-gray-200">{localExp.narrative}</p>
          ) : null}
          <ChartPanel
            height={Math.max(240, shapBars.length * 26)}
            empty={shapBars.length === 0 ? "No SHAP values returned" : null}
            caption="Signed SHAP for this row. Changing inputs and clicking Predict recomputes these values."
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={shapBars}
                layout="vertical"
                margin={{ top: 8, right: 16, left: 8, bottom: 8 }}
              >
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fill: "#8aa0b0", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={140}
                  tick={{ fill: "#f0e6d6", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip content={<DeskTooltip />} />
                <Bar dataKey="shap" name="SHAP (kWh)" maxBarSize={14} isAnimationActive={false}>
                  {shapBars.map((row) => (
                    <Cell key={row.name} fill={row.shap >= 0 ? SERIES.predicted : SERIES.flagged} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
          <table className="desk-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>SHAP</th>
              </tr>
            </thead>
            <tbody>
              {shapBars.map((row) => {
                const src = localExp.features.find(
                  (f) => occupancyFeatureLabel(f.name, localExp.occupancy_source) === row.name,
                );
                return (
                  <tr key={row.name}>
                    <td>{row.name}</td>
                    <td className="tabular-nums">{formatMetric(src?.value)}</td>
                    <td
                      className={`tabular-nums ${row.shap >= 0 ? "text-phosphor" : "text-rust"}`}
                    >
                      {row.shap >= 0 ? "+" : ""}
                      {formatMetric(row.shap)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}

function InputField({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-gray-400">{label}</span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="mt-1 block w-full rounded-md border border-gray-600 bg-gray-900 px-3 py-1.5 text-sm text-gray-100 placeholder-gray-500 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
      />
    </label>
  );
}
