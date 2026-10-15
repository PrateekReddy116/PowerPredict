import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { apiGet, AuthError } from "../api/client";
import type { ModelPerformance } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { ChartPanel } from "../components/ChartPanel";
import { DeskTooltip } from "../components/DeskTooltip";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";
import { PageHeader } from "../components/PageHeader";
import { ChartSkeleton, TableSkeleton } from "../components/Skeleton";
import { SERIES } from "../lib/colors";
import { formatMetric, formatTs } from "../lib/format";
import { MODEL_ORDER, modelLabel } from "../lib/models";

function latestPerModel(rows: ModelPerformance[]): ModelPerformance[] {
  const map = new Map<string, ModelPerformance>();
  const sorted = [...rows].sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""));
  for (const row of sorted) {
    if (!map.has(row.model_name)) map.set(row.model_name, row);
  }
  const ordered: ModelPerformance[] = [];
  for (const name of MODEL_ORDER) {
    const row = map.get(name);
    if (row) ordered.push(row);
  }
  for (const [name, row] of map) {
    if (!(MODEL_ORDER as readonly string[]).includes(name)) ordered.push(row);
  }
  return ordered;
}

export function ModelsPage() {
  const { token, logout } = useAuth();
  const [rows, setRows] = useState<ModelPerformance[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const retry = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!token) return;
    const ac = new AbortController();
    setError(null);
    apiGet<ModelPerformance[]>("/api/model-performance", token, { signal: ac.signal })
      .then(setRows)
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        if (err instanceof AuthError) logout();
        else setError(err instanceof Error ? err.message : "Could not load model performance");
      });
    return () => ac.abort();
  }, [token, logout, nonce]);

  const latest = useMemo(() => (rows ? latestPerModel(rows) : []), [rows]);
  const chartRows = latest.filter((row) => row.mae !== null || row.rmse !== null);
  const smoke = latest.some((row) => row.smoke);

  return (
    <section>
      <PageHeader
        kicker="Evaluation"
        title="Model comparison"
        description="MAE, RMSE, R², and MAPE from /api/model-performance only (database rows, or artifacts/metrics.json if the table is empty). No placeholder scores."
      />
      {error ? <ErrorBanner message={error} onRetry={retry} /> : null}
      {rows === null && !error ? (
        <div className="mt-6 space-y-4">
          <TableSkeleton />
          <ChartSkeleton />
        </div>
      ) : null}
      {rows && rows.length === 0 ? (
        <div className="mt-6">
          <EmptyState
            title="No model runs"
            body="This table stays empty until offline training writes model_metrics or artifacts/metrics.json. Linear regression, random forest, XGBoost, LSTM, and CNN-LSTM appear only when those files exist."
          />
        </div>
      ) : null}
      {rows && rows.length > 0 ? (
        <div className="mt-6 space-y-6">
          {smoke ? (
            <p className="text-sm text-sim">
              At least one row is labeled smoke by the API. Treat those numbers as a pipeline check, not a BDG2 result.
            </p>
          ) : null}
          <div className="overflow-x-auto">
            <table className="desk-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                  <th>R²</th>
                  <th>MAPE</th>
                  <th>Source</th>
                  <th>Stored</th>
                </tr>
              </thead>
              <tbody>
                {latest.map((row) => (
                  <tr key={row.id ?? row.model_name}>
                    <td className="font-display">{modelLabel(row.model_name)}</td>
                    <td className="tabular-nums">{formatMetric(row.mae)}</td>
                    <td className="tabular-nums">{formatMetric(row.rmse)}</td>
                    <td className="tabular-nums">{formatMetric(row.r2)}</td>
                    <td className="tabular-nums">{formatMetric(row.mape)}</td>
                    <td className="font-mono text-xs text-mute">
                      {row.source}
                      {row.smoke ? " · smoke" : ""}
                    </td>
                    <td className="font-mono text-xs text-mute">{formatTs(row.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ChartPanel
            height={300}
            empty={chartRows.length === 0 ? "Stored runs have no numeric MAE/RMSE to plot" : null}
            caption="Bars use only non-null MAE/RMSE from the API. Missing models are omitted, not zero-filled."
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartRows.map((row) => ({
                name: modelLabel(row.model_name),
                MAE: row.mae,
                RMSE: row.rmse,
              }))} margin={{ top: 12, right: 16, left: 8, bottom: 8 }}>
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "#8aa0b0", fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fill: "#8aa0b0", fontSize: 11 }} tickLine={false} axisLine={false} width={48} />
                <Tooltip content={<DeskTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12, color: "#8aa0b0" }} />
                <Bar dataKey="MAE" fill={SERIES.predicted} maxBarSize={28} isAnimationActive={false} />
                <Bar dataKey="RMSE" fill={SERIES.actual} maxBarSize={28} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
        </div>
      ) : null}
    </section>
  );
}
