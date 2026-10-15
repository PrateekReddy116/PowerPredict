import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import { loadEnergyHistory } from "../api/energy";
import { apiGet, AuthError } from "../api/client";
import type { Building, Reading } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { ChartPanel } from "../components/ChartPanel";
import { DeskTooltip } from "../components/DeskTooltip";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";
import { OccupancyCaption } from "../components/OccupancyCaption";
import { OriginBadge } from "../components/OriginBadge";
import { PageHeader } from "../components/PageHeader";
import { SelectField } from "../components/SelectField";
import { ChartSkeleton, TableSkeleton } from "../components/Skeleton";
import { useBuildings } from "../hooks/useBuildings";
import { useSelectedBuilding } from "../hooks/useSelectedBuilding";
import {
  GRAIN_LABELS,
  aggregateEnergy,
  downsample,
  temperatureVsEnergy,
  type Grain,
} from "../lib/aggregate";
import { SERIES } from "../lib/colors";
import { formatNumber, formatTempC, formatTsShort } from "../lib/format";
import { parseOrigin } from "../origin";

const GRAIN_OPTIONS: { value: Grain; label: string }[] = [
  { value: "hourly", label: "Hourly" },
  { value: "daily", label: "Daily" },
  { value: "weekly", label: "Weekly" },
  { value: "monthly", label: "Monthly" },
  { value: "weekday", label: "Weekday mean" },
];

export function BuildingsPage() {
  const { token, logout } = useAuth();
  const { buildings, loading: buildingsLoading, error: buildingsError, retry: retryBuildings } =
    useBuildings();
  const { buildingId, setBuildingId } = useSelectedBuilding(buildings.map((b) => b.id));
  const [grain, setGrain] = useState<Grain>("daily");
  const [detail, setDetail] = useState<Building | null>(null);
  const [history, setHistory] = useState<Reading[] | null>(null);
  const [truncated, setTruncated] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const retry = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!token || !buildingId) {
      setDetail(null);
      setHistory([]);
      return;
    }
    const ac = new AbortController();
    setError(null);
    setHistory(null);
    setDetail(null);
    Promise.all([
      apiGet<Building>(`/api/buildings/${encodeURIComponent(buildingId)}`, token, { signal: ac.signal }),
      loadEnergyHistory(token, buildingId, { signal: ac.signal }),
    ])
      .then(([building, hist]) => {
        setDetail(building);
        setHistory(hist.rows);
        setTruncated(hist.truncated);
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        if (err instanceof AuthError) logout();
        else setError(err instanceof Error ? err.message : "Could not load history");
      });
    return () => ac.abort();
  }, [token, buildingId, logout, nonce]);

  const aggregated = useMemo(() => {
    if (!history) return [];
    return aggregateEnergy(history, grain);
  }, [history, grain]);

  const chartSeries = useMemo(() => {
    const mapped = aggregated.map((row) => ({
      label: grain === "hourly" ? formatTsShort(row.label) : row.label,
      kWh: row.kwh,
    }));
    return downsample(mapped, 2500);
  }, [aggregated, grain]);

  const scatter = useMemo(() => (history ? temperatureVsEnergy(history) : []), [history]);
  const scatterPlot = useMemo(() => downsample(scatter, 1500).items, [scatter]);

  const occupancySource = history?.find((row) => row.occupancy_source)?.occupancy_source;
  const buildingOptions = buildings.map((b) => ({
    value: b.id,
    label: b.primary_use ? `${b.id} · ${b.primary_use}` : b.id,
  }));

  if (!buildingsLoading && buildings.length === 0 && !buildingsError) {
    return (
      <section>
        <PageHeader
          kicker="Archive"
          title="Historical analytics"
          description="Daily, weekly, monthly, hourly, and weekday views derived from /api/energy/history."
        />
        <div className="mt-6">
          <EmptyState
            title="No buildings selected"
            body="Phase 2 ingest inserts building rows from BDG2. Site IDs are not invented here."
          />
        </div>
      </section>
    );
  }

  return (
    <section>
      <PageHeader
        kicker="Archive"
        title="Historical analytics"
        description="The energy API returns hourly rows. Daily / weekly / monthly / weekday views are aggregated in the browser and labeled as such."
        actions={
          <>
            <SelectField
              id="hist-building"
              label="Building"
              value={buildingId ?? ""}
              onChange={setBuildingId}
              options={buildingOptions}
              disabled={buildingsLoading}
            />
            <SelectField
              id="hist-grain"
              label="Grain"
              value={grain}
              onChange={(value) => setGrain(value as Grain)}
              options={GRAIN_OPTIONS}
            />
          </>
        }
      />
      {buildingsError ? <ErrorBanner message={buildingsError} onRetry={retryBuildings} /> : null}
      {error ? <ErrorBanner message={error} onRetry={retry} /> : null}

      {buildingsLoading ? (
        <div className="mt-6">
          <TableSkeleton rows={4} />
        </div>
      ) : (
        <div className="mt-6 overflow-x-auto">
          <table className="desk-table">
            <thead>
              <tr>
                <th>Building</th>
                <th>Site</th>
                <th>Use</th>
                <th>Area</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {buildings.map((row) => (
                <tr key={row.id} className={row.id === buildingId ? "bg-console-2/40" : undefined}>
                  <td className="font-mono text-xs">{row.id}</td>
                  <td>{row.site_id}</td>
                  <td>{row.primary_use ?? "—"}</td>
                  <td className="tabular-nums">{row.sqm ? `${formatNumber(row.sqm, 0)} m²` : "—"}</td>
                  <td className="max-w-sm text-mute">{row.selected_reason ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {detail ? (
        <p className="mt-3 font-mono text-xs text-mute">
          Timezone {detail.timezone}
          {detail.sqm ? ` · ${formatNumber(detail.sqm, 0)} m²` : ""} from /api/buildings/{detail.id}
        </p>
      ) : null}

      {occupancySource ? (
        <div className="mt-3">
          <OccupancyCaption source={occupancySource} />
        </div>
      ) : null}

      <h3 className="mt-8 font-display text-lg font-bold">Energy</h3>
      {history === null ? (
        <div className="mt-3">
          <ChartSkeleton height={300} />
        </div>
      ) : history.length === 0 ? (
        <div className="mt-3">
          <EmptyState
            title="No history for this building"
            body=" /api/energy/history returned no rows. Ingest cleaned BDG2 electricity before aggregating grains."
          />
        </div>
      ) : (
        <div className="mt-3 space-y-6">
          <ChartPanel
            height={300}
            caption={`${GRAIN_LABELS[grain]}. ${history.length} hourly rows from /api/energy/history${
              truncated ? " (window truncated at 3×10,000 API pages)" : ""
            }.${chartSeries.factor > 1 ? ` Chart downsampled ×${chartSeries.factor} for rendering.` : ""}`}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartSeries.items} margin={{ top: 12, right: 16, left: 8, bottom: 8 }}>
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" vertical={false} />
                <XAxis dataKey="label" tick={{ fill: "#8aa0b0", fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fill: "#8aa0b0", fontSize: 11 }} tickLine={false} axisLine={false} width={56} />
                <Tooltip content={<DeskTooltip valueSuffix="kWh" />} />
                <Line
                  type="monotone"
                  dataKey="kWh"
                  stroke={SERIES.actual}
                  dot={false}
                  strokeWidth={1.6}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
          <p className="flex flex-wrap items-center gap-2 text-[11px] text-mute">
            Series origins in this window:
            {[...new Set(history.map((row) => row.data_origin))].map((origin) => {
              const parsed = parseOrigin(origin);
              return parsed ? <OriginBadge key={origin} origin={parsed} /> : null;
            })}
          </p>
        </div>
      )}

      <h3 className="mt-8 font-display text-lg font-bold">Temperature vs energy</h3>
      {history === null ? (
        <div className="mt-3">
          <ChartSkeleton />
        </div>
      ) : scatter.length === 0 ? (
        <div className="mt-3">
          <EmptyState
            title="No temperature in this history"
            body="The scatter uses temperature_c on /api/energy/history rows. If that field is null, the plot stays empty."
          />
        </div>
      ) : (
        <div className="mt-3">
          <ChartPanel
            height={300}
            caption={`Hourly pairs with temperature_c (${scatter.length} points). ${formatTempC(
              scatter[0]?.temperature_c,
            )} shown as °C vs kWh.`}
          >
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 12, right: 16, left: 8, bottom: 8 }}>
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" />
                <XAxis
                  dataKey="temperature_c"
                  name="Temperature"
                  unit="°C"
                  tick={{ fill: "#8aa0b0", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  dataKey="energy_kwh"
                  name="Energy"
                  unit=" kWh"
                  tick={{ fill: "#8aa0b0", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={56}
                />
                <ZAxis range={[20, 20]} />
                <Tooltip content={<DeskTooltip />} />
                <Scatter data={scatterPlot} fill={SERIES.predicted} isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </ChartPanel>
        </div>
      )}
    </section>
  );
}
