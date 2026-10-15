import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { apiGet, AuthError, withQuery } from "../api/client";
import type { Forecast, ModelPerformance } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { ChartPanel } from "../components/ChartPanel";
import { DeskTooltip } from "../components/DeskTooltip";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";
import { KpiTile } from "../components/KpiTile";
import { PageHeader } from "../components/PageHeader";
import { SelectField } from "../components/SelectField";
import { ChartSkeleton } from "../components/Skeleton";
import { useBuildings } from "../hooks/useBuildings";
import { useSelectedBuilding } from "../hooks/useSelectedBuilding";
import { MODEL_SELECT_OPTIONS, useSelectedModel } from "../hooks/useSelectedModel";
import { SERIES } from "../lib/colors";
import { downsample } from "../lib/aggregate";
import { formatMetric, formatTsShort } from "../lib/format";
import { modelLabel } from "../lib/models";

export function ForecastPage() {
  const { token, logout } = useAuth();
  const { buildings, loading: buildingsLoading, error: buildingsError, retry: retryBuildings } =
    useBuildings();
  const { buildingId, setBuildingId } = useSelectedBuilding(buildings.map((b) => b.id));
  const { model, setModel } = useSelectedModel();

  const [series, setSeries] = useState<Forecast[] | null>(null);
  const [performance, setPerformance] = useState<ModelPerformance[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [perfError, setPerfError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);
  const retry = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!token) return;
    const ac = new AbortController();
    setError(null);
    setPerfError(null);
    apiGet<ModelPerformance[]>("/api/model-performance", token, { signal: ac.signal })
      .then(setPerformance)
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        if (err instanceof AuthError) logout();
        else {
          setPerformance([]);
          setPerfError(err instanceof Error ? err.message : "Could not load model performance");
        }
      });
    return () => ac.abort();
  }, [token, logout, nonce]);

  useEffect(() => {
    if (!token || !buildingId) {
      setSeries([]);
      return;
    }
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    setSeries(null);
    apiGet<Forecast[]>(
      withQuery("/api/forecast", { building_id: buildingId, model }),
      token,
      { signal: ac.signal },
    )
      .then(setSeries)
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        if (err instanceof AuthError) logout();
        else setError(err instanceof Error ? err.message : "Could not load forecasts");
      })
      .finally(() => {
        if (!ac.signal.aborted) setLoading(false);
      });
    return () => ac.abort();
  }, [token, buildingId, model, logout, nonce]);

  const metrics = useMemo(() => {
    if (!performance) return null;
    const matches = performance.filter((row) => row.model_name === model);
    if (matches.length === 0) return null;
    return matches.sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""))[0] ?? null;
  }, [performance, model]);

  const chartData = useMemo(() => {
    if (!series) return [];
    const mapped = series.map((row) => ({
      ts: formatTsShort(row.target_ts),
      actual: row.y_true,
      predicted: row.y_pred,
    }));
    return downsample(mapped, 2500).items;
  }, [series]);

  const buildingOptions = buildings.map((b) => ({
    value: b.id,
    label: b.primary_use ? `${b.id} · ${b.primary_use}` : b.id,
  }));

  if (!buildingsLoading && buildings.length === 0 && !buildingsError) {
    return (
      <section>
        <PageHeader
          kicker="Forecasting"
          title="Actual vs predicted"
          description="Next-hour electricity. Plots stored y_true / y_pred only."
        />
        <div className="mt-6">
          <EmptyState
            title="No buildings in catalog"
            body="Forecasts are keyed by building_id. Ingest BDG2 before this page can plot a series."
          />
        </div>
      </section>
    );
  }

  return (
    <section>
      <PageHeader
        kicker="Forecasting"
        title="Actual vs predicted"
        description="Stored next-hour y_pred from offline training. The forecast API has no horizon parameter, so this desk does not show a horizon control."
        actions={
          <>
            <SelectField
              id="forecast-building"
              label="Building"
              value={buildingId ?? ""}
              onChange={setBuildingId}
              options={buildingOptions}
              disabled={buildingsLoading}
            />
            <SelectField
              id="forecast-model"
              label="Model"
              value={model}
              onChange={setModel}
              options={MODEL_SELECT_OPTIONS}
            />
          </>
        }
      />
      {buildingsError ? <ErrorBanner message={buildingsError} onRetry={retryBuildings} /> : null}
      {perfError ? <ErrorBanner message={perfError} onRetry={retry} /> : null}
      {error ? <ErrorBanner message={error} onRetry={retry} /> : null}

      <div className="mt-5 grid grid-cols-2 gap-px bg-hair sm:grid-cols-4">
        <KpiTile label="MAE" value={formatMetric(metrics?.mae)} hint={metrics ? `Source: ${metrics.source}` : "No row for this model"} loading={!performance} />
        <KpiTile label="RMSE" value={formatMetric(metrics?.rmse)} hint={metrics?.smoke ? "API labeled this a smoke run" : "From /api/model-performance"} loading={!performance} />
        <KpiTile label="R²" value={formatMetric(metrics?.r2)} loading={!performance} />
        <KpiTile label="MAPE" value={metrics?.mape === null || metrics?.mape === undefined ? "—" : formatMetric(metrics.mape)} loading={!performance} />
      </div>
      {metrics?.smoke ? (
        <p className="mt-3 text-sm text-sim">This performance row is labeled smoke by the API — not a BDG2 evaluation.</p>
      ) : null}

      <div className="mt-6">
        {loading && !series ? (
          <ChartSkeleton height={320} />
        ) : (
          <ChartPanel
            height={320}
            empty={!series || series.length === 0 ? `No stored forecasts for ${modelLabel(model)}` : null}
            caption={`${modelLabel(model)} · ${series?.length ?? 0} points from /api/forecast. X-axis is the feature timestamp t (target is the following hour). Training is an offline CLI.`}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 12, right: 16, left: 8, bottom: 8 }}>
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" vertical={false} />
                <XAxis dataKey="ts" tick={{ fill: "#8aa0b0", fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fill: "#8aa0b0", fontSize: 11 }} tickLine={false} axisLine={false} width={48} />
                <Tooltip content={<DeskTooltip valueSuffix="kWh" />} />
                <Legend wrapperStyle={{ fontSize: 12, color: "#8aa0b0" }} />
                <Line
                  type="monotone"
                  dataKey="actual"
                  name="Actual"
                  stroke={SERIES.actual}
                  dot={false}
                  strokeWidth={1.6}
                  connectNulls={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  name="Predicted"
                  stroke={SERIES.predicted}
                  dot={false}
                  strokeWidth={1.6}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}
      </div>
    </section>
  );
}
