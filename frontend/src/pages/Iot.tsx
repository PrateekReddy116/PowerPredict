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

import { loadEnergyHistory } from "../api/energy";
import { apiGet, AuthError, withQuery } from "../api/client";
import type { IotStatus, Reading } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { ChartPanel } from "../components/ChartPanel";
import { DeskTooltip } from "../components/DeskTooltip";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";
import { IotStatusChip } from "../components/IotStatusChip";
import { OccupancyCaption } from "../components/OccupancyCaption";
import { OriginBadge } from "../components/OriginBadge";
import { PageHeader } from "../components/PageHeader";
import { SelectField } from "../components/SelectField";
import { ChartSkeleton, TableSkeleton } from "../components/Skeleton";
import { useBuildings } from "../hooks/useBuildings";
import { useInterval } from "../hooks/useInterval";
import { useSelectedBuilding } from "../hooks/useSelectedBuilding";
import { downsample } from "../lib/aggregate";
import { SERIES } from "../lib/colors";
import { formatHumidity, formatKwh, formatOccupancy, formatTempC, formatTs, formatTsShort } from "../lib/format";
import { occupancyCaption } from "../lib/occupancy";

const POLL_MS = 5000;

export function IotPage() {
  const { token, logout } = useAuth();
  const { buildings, loading: buildingsLoading, error: buildingsError, retry: retryBuildings } =
    useBuildings();
  const { buildingId, setBuildingId } = useSelectedBuilding(buildings.map((b) => b.id));
  const [status, setStatus] = useState<IotStatus | null>(null);
  const [latest, setLatest] = useState<Reading[] | null>(null);
  const [simSeries, setSimSeries] = useState<Reading[]>([]);
  const [liveSeries, setLiveSeries] = useState<Reading[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const retry = useCallback(() => setNonce((n) => n + 1), []);

  const poll = useCallback(() => {
    if (!token) return;
    const ac = new AbortController();
    const fail = (err: unknown) => {
      if (ac.signal.aborted) return;
      if (err instanceof AuthError) logout();
      else setError(err instanceof Error ? err.message : "Could not load IoT");
    };

    apiGet<IotStatus>("/api/iot/status", token, { signal: ac.signal })
      .then((st) => {
        if (!ac.signal.aborted) setStatus(st);
      })
      .catch(fail);

    apiGet<Reading[]>(withQuery("/api/iot/latest", { building_id: buildingId }), token, {
      signal: ac.signal,
    })
      .then((rows) => {
        if (ac.signal.aborted) return;
        setLatest(rows);
        setError(null);
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        setLatest([]);
        fail(err);
      });

    if (buildingId) {
      Promise.all([
        loadEnergyHistory(token, buildingId, { origin: "simulated_iot", signal: ac.signal, maxPages: 1 }),
        loadEnergyHistory(token, buildingId, { origin: "real_iot", signal: ac.signal, maxPages: 1 }),
      ])
        .then(([sim, live]) => {
          if (ac.signal.aborted) return;
          setSimSeries(sim.rows);
          setLiveSeries(live.rows);
        })
        .catch(fail);
    } else {
      setSimSeries([]);
      setLiveSeries([]);
    }

    return () => ac.abort();
  }, [token, buildingId, logout]);

  useEffect(() => {
    setSimSeries([]);
    setLiveSeries([]);
    setLatest(null);
  }, [buildingId]);

  useEffect(() => {
    const cancel = poll();
    return cancel;
  }, [poll, nonce]);

  useInterval(poll, token ? POLL_MS : null);

  const chartData = useMemo(() => {
    const byTs = new Map<string, { ts: string; simulated?: number; live?: number }>();
    for (const row of simSeries) {
      byTs.set(row.ts, { ...(byTs.get(row.ts) ?? { ts: formatTsShort(row.ts) }), simulated: row.energy_kwh, ts: formatTsShort(row.ts) });
    }
    for (const row of liveSeries) {
      byTs.set(row.ts, { ...(byTs.get(row.ts) ?? { ts: formatTsShort(row.ts) }), live: row.energy_kwh, ts: formatTsShort(row.ts) });
    }
    const merged = [...byTs.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([, value]) => value);
    return downsample(merged, 1500).items;
  }, [simSeries, liveSeries]);

  const buildingOptions = buildings.map((b) => ({
    value: b.id,
    label: b.primary_use ? `${b.id} · ${b.primary_use}` : b.id,
  }));

  const occupancySource = latest?.find((row) => row.occupancy_source)?.occupancy_source;

  return (
    <section>
      <PageHeader
        kicker="Telemetry"
        title="IoT monitoring"
        description="Polled every 5 seconds. SIMULATED is the honest demo label for the MQTT publisher. LIVE is reserved for real_iot and is never used for the simulator."
        actions={
          <SelectField
            id="iot-building"
            label="Building"
            value={buildingId ?? ""}
            onChange={(value) => {
              if (value) setBuildingId(value);
            }}
            options={buildingOptions}
            disabled={buildingsLoading}
          />
        }
      />
      {buildingsError ? <ErrorBanner message={buildingsError} onRetry={retryBuildings} /> : null}
      {error ? <ErrorBanner message={error} onRetry={retry} /> : null}

      <div className="mt-5 border border-hair bg-console/50 px-4 py-4">
        <div className="flex flex-wrap items-center gap-3">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-mute">Connection</p>
          {status ? <IotStatusChip status={status.status} /> : <span className="text-sm text-mute">Loading…</span>}
        </div>
        {status ? (
          <dl className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
            <div>
              <dt className="font-mono text-[10px] uppercase tracking-[0.14em] text-mute">Broker</dt>
              <dd>{status.broker_connected ? "Connected" : "Unreachable"}</dd>
            </div>
            <div>
              <dt className="font-mono text-[10px] uppercase tracking-[0.14em] text-mute">Last message</dt>
              <dd className="font-mono text-xs">{formatTs(status.last_message_at)}</dd>
            </div>
            <div>
              <dt className="font-mono text-[10px] uppercase tracking-[0.14em] text-mute">Topic</dt>
              <dd className="font-mono text-xs">{status.topic}</dd>
            </div>
            <div>
              <dt className="font-mono text-[10px] uppercase tracking-[0.14em] text-mute">Last origin</dt>
              <dd>
                {status.last_data_origin ? <OriginBadge origin={status.last_data_origin} /> : "—"}
              </dd>
            </div>
          </dl>
        ) : null}
        {status ? <p className="mt-3 max-w-3xl text-[12px] leading-5 text-mute">{status.note}</p> : null}
      </div>

      {occupancySource ? (
        <div className="mt-4">
          <OccupancyCaption source={occupancySource} />
        </div>
      ) : null}

      <h3 className="mt-8 font-display text-lg font-bold">Latest sensors</h3>
      {latest === null ? (
        <div className="mt-3">
          <TableSkeleton rows={3} />
        </div>
      ) : latest.length === 0 ? (
        <div className="mt-3">
          <EmptyState
            title="No MQTT readings stored"
            body="The simulator publishes data_origin=simulated_iot. Historical CSV ingest never appears here. Start `python -m iot.simulator` when Mosquitto is up."
          />
        </div>
      ) : (
        <div className="mt-3 overflow-x-auto">
          <table className="desk-table">
            <thead>
              <tr>
                <th>Building</th>
                <th>Origin</th>
                <th>Time</th>
                <th>Energy</th>
                <th>Temp</th>
                <th>Humidity</th>
                <th>Occupancy</th>
              </tr>
            </thead>
            <tbody>
              {latest.map((row) => (
                <tr key={row.id}>
                  <td className="font-mono text-xs">{row.building_id}</td>
                  <td>
                    <OriginBadge origin={row.data_origin} />
                  </td>
                  <td className="font-mono text-xs">{formatTs(row.ts)}</td>
                  <td className="tabular-nums">{formatKwh(row.energy_kwh)}</td>
                  <td className="tabular-nums">{formatTempC(row.temperature_c)}</td>
                  <td className="tabular-nums">{formatHumidity(row.humidity_pct)}</td>
                  <td className="tabular-nums" title={occupancyCaption(row.occupancy_source)}>
                    {formatOccupancy(row.occupancy)}
                    {row.occupancy_source === "simulated" ? (
                      <span className="ml-2 font-mono text-[10px] uppercase text-sim">simulated</span>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <h3 className="mt-8 font-display text-lg font-bold">IoT energy series</h3>
      <div className="mt-3">
        {latest === null ? (
          <ChartSkeleton />
        ) : (
          <ChartPanel
            height={300}
            empty={chartData.length === 0 ? "No simulated_iot or real_iot history for this building" : null}
            caption="Separate traces for MQTT simulator and live meters. Polling only — no WebSocket."
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
                  dataKey="simulated"
                  name="MQTT simulator"
                  stroke={SERIES.simulated}
                  strokeDasharray="4 4"
                  dot={false}
                  strokeWidth={1.6}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="live"
                  name="Live meter"
                  stroke={SERIES.live}
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
