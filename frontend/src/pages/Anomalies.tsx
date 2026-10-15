import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { apiGet, AuthError, withQuery } from "../api/client";
import type { Anomaly } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { ChartPanel } from "../components/ChartPanel";
import { DeskTooltip } from "../components/DeskTooltip";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";
import { OriginBadge } from "../components/OriginBadge";
import { PageHeader } from "../components/PageHeader";
import { SelectField } from "../components/SelectField";
import { ChartSkeleton, TableSkeleton } from "../components/Skeleton";
import { useBuildings } from "../hooks/useBuildings";
import { useSelectedBuilding } from "../hooks/useSelectedBuilding";
import { downsample } from "../lib/aggregate";
import { SERIES } from "../lib/colors";
import { formatKwh, formatMetric, formatPct, formatTs, formatTsShort } from "../lib/format";
import { ORIGIN_LABELS, type DataOrigin } from "../origin";

const ORIGIN_OPTIONS: { value: string; label: string }[] = [
  { value: "all", label: "All origins" },
  { value: "historical", label: ORIGIN_LABELS.historical },
  { value: "simulated_iot", label: ORIGIN_LABELS.simulated_iot },
  { value: "real_iot", label: ORIGIN_LABELS.real_iot },
];

const SEVERITY_OPTIONS = [
  { value: "all", label: "All severities" },
  { value: "flagged", label: "Flagged only" },
  { value: "warning", label: "Warning" },
  { value: "critical", label: "Critical" },
  { value: "normal", label: "Normal (not flagged)" },
];

function severityClass(severity: string | null): string {
  if (severity === "critical") return "text-rust";
  if (severity === "warning") return "text-sim";
  return "text-mute";
}

export function AnomaliesPage() {
  const { token, logout } = useAuth();
  const { buildings, loading: buildingsLoading, error: buildingsError, retry: retryBuildings } =
    useBuildings();
  const { buildingId, setBuildingId } = useSelectedBuilding(buildings.map((b) => b.id));
  const [origin, setOrigin] = useState("all");
  const [severity, setSeverity] = useState("flagged");
  const [rows, setRows] = useState<Anomaly[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const retry = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!token || !buildingId) {
      setRows([]);
      return;
    }
    const ac = new AbortController();
    setError(null);
    setRows(null);
    apiGet<Anomaly[]>(
      withQuery("/api/anomalies", {
        building_id: buildingId,
        origin: origin === "all" ? undefined : origin,
      }),
      token,
      { signal: ac.signal },
    )
      .then(setRows)
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        if (err instanceof AuthError) logout();
        else setError(err instanceof Error ? err.message : "Could not load anomalies");
      });
    return () => ac.abort();
  }, [token, buildingId, origin, logout, nonce]);

  const filtered = useMemo(() => {
    if (!rows) return [];
    return rows.filter((row) => {
      if (severity === "flagged") return row.is_anomaly;
      if (severity === "all") return true;
      return (row.severity ?? "") === severity;
    });
  }, [rows, severity]);

  const chartData = useMemo(() => {
    const mapped = filtered.map((row) => ({
      ts: formatTsShort(row.ts),
      actual: row.actual_kwh,
      expected: row.expected_kwh,
      score: row.score,
      flagged: row.is_anomaly ? row.actual_kwh : null,
    }));
    return downsample(mapped, 2500).items;
  }, [filtered]);

  const buildingOptions = buildings.map((b) => ({
    value: b.id,
    label: b.primary_use ? `${b.id} · ${b.primary_use}` : b.id,
  }));

  const emptyCatalog = !buildingsLoading && buildings.length === 0 && !buildingsError;

  return (
    <section>
      <PageHeader
        kicker="Detection"
        title="Anomalies"
        description="Isolation Forest scores persisted from the chronological train window. Severity comes from the API, not hardcoded UI thresholds."
        actions={
          <>
            <SelectField
              id="anomaly-building"
              label="Building"
              value={buildingId ?? ""}
              onChange={setBuildingId}
              options={buildingOptions}
              disabled={buildingsLoading}
            />
            <SelectField id="anomaly-origin" label="Origin" value={origin} onChange={setOrigin} options={ORIGIN_OPTIONS} />
            <SelectField
              id="anomaly-severity"
              label="Severity"
              value={severity}
              onChange={setSeverity}
              options={SEVERITY_OPTIONS}
            />
          </>
        }
      />
      {buildingsError ? <ErrorBanner message={buildingsError} onRetry={retryBuildings} /> : null}
      {error ? <ErrorBanner message={error} onRetry={retry} /> : null}
      {emptyCatalog ? (
        <div className="mt-6">
          <EmptyState
            title="No buildings in catalog"
            body="Anomaly rows are stored per building after Isolation Forest training. Nothing is listed until ingest exists."
          />
        </div>
      ) : null}
      {rows === null && !error && !emptyCatalog ? (
        <div className="mt-6 space-y-4">
          <ChartSkeleton />
          <TableSkeleton />
        </div>
      ) : null}
      {rows && rows.length === 0 && !emptyCatalog ? (
        <div className="mt-6">
          <EmptyState
            title="No anomaly scores"
            body="This list stays empty until Isolation Forest writes the anomalies table. Counts are never invented."
          />
        </div>
      ) : null}
      {rows && rows.length > 0 ? (
        <div className="mt-6 space-y-6">
          <p className="font-mono text-xs text-mute">
            {filtered.length} shown of {rows.length} returned for {buildingId}
            {origin !== "all" ? ` · ${ORIGIN_LABELS[origin as DataOrigin]}` : ""}.
          </p>
          <ChartPanel
            height={300}
            empty={filtered.length === 0 ? "No rows match the current filters" : null}
            caption="Actual vs expected kWh. Flagged hours are marked on the actual series."
          >
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 12, right: 16, left: 8, bottom: 8 }}>
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" vertical={false} />
                <XAxis dataKey="ts" tick={{ fill: "#8aa0b0", fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fill: "#8aa0b0", fontSize: 11 }} tickLine={false} axisLine={false} width={48} />
                <Tooltip content={<DeskTooltip valueSuffix="kWh" />} />
                <Legend wrapperStyle={{ fontSize: 12, color: "#8aa0b0" }} />
                <Line type="monotone" dataKey="actual" name="Actual" stroke={SERIES.actual} dot={false} strokeWidth={1.5} isAnimationActive={false} />
                <Line type="monotone" dataKey="expected" name="Expected" stroke={SERIES.expected} dot={false} strokeWidth={1.5} isAnimationActive={false} />
                <Scatter dataKey="flagged" name="Flagged" fill={SERIES.flagged} />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartPanel>
          <div className="overflow-x-auto">
            <table className="desk-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Origin</th>
                  <th>Actual</th>
                  <th>Expected</th>
                  <th>Dev</th>
                  <th>Score</th>
                  <th>Severity</th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(-200).reverse().map((row) => (
                  <tr key={row.id}>
                    <td className="font-mono text-xs">{formatTs(row.ts)}</td>
                    <td>
                      <OriginBadge origin={row.data_origin} />
                    </td>
                    <td className="tabular-nums">{formatKwh(row.actual_kwh)}</td>
                    <td className="tabular-nums">{formatKwh(row.expected_kwh)}</td>
                    <td className="tabular-nums">{formatPct(row.deviation_pct)}</td>
                    <td className="tabular-nums">{formatMetric(row.score)}</td>
                    <td className={`font-mono text-xs uppercase ${severityClass(row.severity)}`}>
                      {row.severity ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {filtered.length > 200 ? (
            <p className="text-[11px] text-mute">Table shows the 200 most recent matching rows.</p>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
