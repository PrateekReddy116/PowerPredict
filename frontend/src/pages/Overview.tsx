import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { loadEnergyHistory } from "../api/energy";
import { apiGet, AuthError, withQuery } from "../api/client";
import type { Anomaly, Forecast, IotStatus, Reading } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { ChartPanel } from "../components/ChartPanel";
import { DeskTooltip } from "../components/DeskTooltip";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";
import { IotStatusChip } from "../components/IotStatusChip";
import { KpiTile } from "../components/KpiTile";
import { OccupancyCaption } from "../components/OccupancyCaption";
import { OriginBadge } from "../components/OriginBadge";
import { PageHeader } from "../components/PageHeader";
import { SelectField } from "../components/SelectField";
import { ChartSkeleton } from "../components/Skeleton";
import { useBuildings } from "../hooks/useBuildings";
import { useSelectedBuilding } from "../hooks/useSelectedBuilding";
import { MODEL_SELECT_OPTIONS, useSelectedModel } from "../hooks/useSelectedModel";
import { SERIES } from "../lib/colors";
import { formatHumidity, formatKwh, formatOccupancy, formatSignedKwh, formatTempC, formatTs, formatTsShort } from "../lib/format";
import { modelLabel } from "../lib/models";
import { occupancyCaption } from "../lib/occupancy";

function latestReading(history: Reading[], iotLatest: Reading[], buildingId: string): Reading | null {
  const histLast = history.at(-1) ?? null;
  const iotLast = iotLatest
    .filter((row) => row.building_id === buildingId)
    .reduce<Reading | null>((best, row) => {
      if (!best) return row;
      return Date.parse(row.ts) > Date.parse(best.ts) ? row : best;
    }, null);
  if (iotLast && histLast) {
    return Date.parse(iotLast.ts) >= Date.parse(histLast.ts) ? iotLast : histLast;
  }
  return iotLast ?? histLast;
}

export function OverviewPage() {
  const { token, logout } = useAuth();
  const { buildings, loading: buildingsLoading, error: buildingsError, retry: retryBuildings } =
    useBuildings();
  const { buildingId, setBuildingId } = useSelectedBuilding(buildings.map((b) => b.id));
  const { model, setModel } = useSelectedModel();

  const [history, setHistory] = useState<Reading[] | null>(null);
  const [truncated, setTruncated] = useState(false);
  const [forecasts, setForecasts] = useState<Forecast[] | null>(null);
  const [iot, setIot] = useState<IotStatus | null>(null);
  const [iotLatest, setIotLatest] = useState<Reading[] | null>(null);
  const [anomalies, setAnomalies] = useState<Anomaly[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);

  const retry = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!token || !buildingId) {
      setHistory([]);
      setForecasts([]);
      setAnomalies([]);
      setIotLatest([]);
      return;
    }
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    setHistory(null);
    setForecasts(null);
    setIotLatest(null);
    setAnomalies(null);
    Promise.all([
      loadEnergyHistory(token, buildingId, { signal: ac.signal }),
      apiGet<Forecast[]>(withQuery("/api/forecast", { building_id: buildingId, model }), token, {
        signal: ac.signal,
      }),
      apiGet<IotStatus>("/api/iot/status", token, { signal: ac.signal }),
      apiGet<Reading[]>(withQuery("/api/iot/latest", { building_id: buildingId }), token, {
        signal: ac.signal,
      }),
      apiGet<Anomaly[]>(withQuery("/api/anomalies", { building_id: buildingId }), token, {
        signal: ac.signal,
      }),
    ])
      .then(([hist, fc, status, latest, flags]) => {
        setHistory(hist.rows);
        setTruncated(hist.truncated);
        setForecasts(fc);
        setIot(status);
        setIotLatest(latest);
        setAnomalies(flags);
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        if (err instanceof AuthError) logout();
        else setError(err instanceof Error ? err.message : "Could not load overview");
      })
      .finally(() => {
        if (!ac.signal.aborted) setLoading(false);
      });
    return () => ac.abort();
  }, [token, buildingId, model, logout, nonce]);

  const current = useMemo(
    () => (history && iotLatest && buildingId ? latestReading(history, iotLatest, buildingId) : null),
    [history, iotLatest, buildingId],
  );

  const forecastView = useMemo(() => {
    if (!forecasts || forecasts.length === 0) return null;
    const sorted = [...forecasts].sort((a, b) => a.target_ts.localeCompare(b.target_ts));
    const next = sorted[sorted.length - 1];
    if (!next) return null;
    // Delta vs latest stored reading (actual), not vs the previous y_pred row —
    // consecutive predictions are often nearly identical and read as 0.0.
    const change =
      current && Number.isFinite(current.energy_kwh)
        ? next.y_pred - current.energy_kwh
        : null;
    return { next, change };
  }, [forecasts, current]);

  const flagged = anomalies?.filter((row) => row.is_anomaly).length ?? null;
  const spark = useMemo(() => {
    if (!history) return [];
    return history.slice(-72).map((row) => ({
      ts: formatTsShort(row.ts),
      kWh: row.energy_kwh,
    }));
  }, [history]);

  const buildingOptions = buildings.map((b) => ({
    value: b.id,
    label: b.primary_use ? `${b.id} · ${b.primary_use}` : b.id,
  }));

  if (!buildingsLoading && buildings.length === 0 && !buildingsError) {
    return (
      <section>
        <PageHeader
          kicker="Operations"
          title="Overview"
          description="Demand, stored next-hour forecasts, weather, occupancy, IoT, and anomaly flags for the selected building."
        />
        <div className="mt-6">
          <EmptyState
            title="No buildings in catalog"
            body="Ingest BDG2 with `python -m ml ingest` so building rows exist. Tiles stay blank until then — values are never invented."
          />
        </div>
      </section>
    );
  }

  return (
    <section>
      <PageHeader
        kicker="Operations"
        title="Overview"
        description="Latest stored reading, next-hour forecast from the API, and IoT connection state. Simulated streams are never labeled LIVE."
        actions={
          <>
            <SelectField
              id="overview-building"
              label="Building"
              value={buildingId ?? ""}
              onChange={setBuildingId}
              options={buildingOptions}
              disabled={buildingsLoading}
            />
            <SelectField
              id="overview-model"
              label="Model"
              value={model}
              onChange={setModel}
              options={MODEL_SELECT_OPTIONS}
            />
          </>
        }
      />
      {buildingsError ? <ErrorBanner message={buildingsError} onRetry={retryBuildings} /> : null}
      {error ? <ErrorBanner message={error} onRetry={retry} /> : null}

      <div className="mt-5 grid grid-cols-2 gap-px bg-hair lg:grid-cols-4">
        <KpiTile
          label="Current energy"
          value={formatKwh(current?.energy_kwh)}
          hint={current ? formatTs(current.ts) : "No reading in the returned history window"}
          badge={current ? <OriginBadge origin={current.data_origin} /> : undefined}
          loading={loading}
        />
        <KpiTile
          label="Next-hour forecast"
          value={formatKwh(forecastView?.next.y_pred)}
          hint={
            forecastView
              ? `Origin ${formatTs(forecastView.next.target_ts)} (features at t → y at t+1) · split ${forecastView.next.split}${
                  forecastView.next.model_name ? ` · ${forecastView.next.model_name}` : ` · ${modelLabel(model)}`
                }`
              : `No stored y_pred for ${modelLabel(model)}`
          }
          loading={loading}
        />
        <KpiTile
          label="Forecast change"
          value={formatSignedKwh(forecastView?.change ?? null)}
          hint={
            forecastView === null
              ? `No stored y_pred for ${modelLabel(model)}`
              : forecastView.change === null
                ? "Needs a current reading and a next-hour forecast"
                : "Next-hour forecast minus current energy (y_pred − latest reading)"
          }
          loading={loading}
        />
        <KpiTile
          label="IoT"
          value={iot?.status ?? "—"}
          hint={iot?.note ?? "Connection state from /api/iot/status"}
          badge={iot ? <IotStatusChip status={iot.status} /> : undefined}
          loading={loading}
        />
        <KpiTile
          label="Temperature"
          value={formatTempC(current?.temperature_c)}
          hint="From the same reading as current energy"
          loading={loading}
        />
        <KpiTile
          label="Humidity"
          value={formatHumidity(current?.humidity_pct)}
          hint="From the same reading as current energy"
          loading={loading}
        />
        <KpiTile
          label="Occupancy"
          value={formatOccupancy(current?.occupancy)}
          hint={current ? occupancyCaption(current.occupancy_source) : "No occupancy on the latest reading"}
          loading={loading}
        />
        <KpiTile
          label="Anomaly flags"
          value={flagged === null ? "—" : String(flagged)}
          hint={
            anomalies
              ? `${flagged} flagged of ${anomalies.length} scored rows returned`
              : "Isolation Forest rows from /api/anomalies"
          }
          loading={loading}
        />
      </div>

      {current ? (
        <div className="mt-4">
          <OccupancyCaption source={current.occupancy_source} />
        </div>
      ) : null}

      <div className="mt-6">
        {loading && !history ? (
          <ChartSkeleton />
        ) : (
          <ChartPanel
            empty={spark.length === 0 ? "No hourly series in the returned history window" : null}
            caption={
              truncated
                ? "Chart shows the last 72 points of the paged history window (API cap 10,000 rows per page)."
                : "Last 72 hourly points from /api/energy/history."
            }
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={spark} margin={{ top: 12, right: 16, left: 8, bottom: 8 }}>
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" vertical={false} />
                <XAxis dataKey="ts" tick={{ fill: "#8aa0b0", fontSize: 11 }} tickLine={false} axisLine={false} />
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
                  stroke={SERIES.actual}
                  dot={false}
                  strokeWidth={1.75}
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
