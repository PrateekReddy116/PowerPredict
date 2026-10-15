import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { apiGetOptional, AuthError, withQuery } from "../api/client";
import type { GlobalExplanation, LocalExplanation } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { ChartPanel } from "../components/ChartPanel";
import { DeskTooltip } from "../components/DeskTooltip";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";
import { OccupancyCaption } from "../components/OccupancyCaption";
import { PageHeader } from "../components/PageHeader";
import { SelectField } from "../components/SelectField";
import { ChartSkeleton, TableSkeleton } from "../components/Skeleton";
import { MODEL_SELECT_OPTIONS, useSelectedModel } from "../hooks/useSelectedModel";
import { SERIES } from "../lib/colors";
import { formatKwh, formatMetric, formatTs } from "../lib/format";
import { modelLabel } from "../lib/models";
import { occupancyFeatureLabel } from "../lib/occupancy";

export function ExplainPage() {
  const { token, logout } = useAuth();
  const { model, setModel } = useSelectedModel();
  const [globalExp, setGlobalExp] = useState<GlobalExplanation | null | undefined>(undefined);
  const [localExp, setLocalExp] = useState<LocalExplanation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const retry = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!token) return;
    const ac = new AbortController();
    setGlobalExp(undefined);
    setLocalExp(null);
    setError(null);
    (async () => {
      try {
        const globalPayload = await apiGetOptional<GlobalExplanation>(
          withQuery("/api/explainability", { model }),
          token,
          { signal: ac.signal },
        );
        if (ac.signal.aborted) return;
        setGlobalExp(globalPayload);
        if (!globalPayload) {
          setLocalExp(null);
          return;
        }
        const localPayload = await apiGetOptional<LocalExplanation>(
          withQuery("/api/explainability", { model, row: 0 }),
          token,
          { signal: ac.signal },
        );
        if (!ac.signal.aborted) setLocalExp(localPayload);
      } catch (err: unknown) {
        if (ac.signal.aborted) return;
        if (err instanceof AuthError) logout();
        else setError(err instanceof Error ? err.message : "Could not load explanations");
      }
    })();
    return () => ac.abort();
  }, [token, model, logout, nonce]);

  const ranking = useMemo(() => {
    if (!globalExp) return [];
    return [...globalExp.ranking]
      .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)
      .slice(0, 15)
      .map((row) => ({
        name: occupancyFeatureLabel(row.name, globalExp.occupancy_source),
        shap: row.mean_abs_shap,
      }));
  }, [globalExp]);

  const occupancySource = globalExp?.occupancy_source ?? localExp?.occupancy_source;
  const loading = globalExp === undefined && !error;

  return (
    <section>
      <PageHeader
        kicker="Explainable AI"
        title="SHAP importance"
        description="Global Feature Importance is mean |SHAP| from the last `python -m ml.explain` run (stored JSON). The local table below is a stored CLI example, not a live What-if explanation. Missing artifacts return 404 — this page stays empty rather than inventing contributions."
        actions={
          <SelectField
            id="explain-model"
            label="Model"
            value={model}
            onChange={setModel}
            options={MODEL_SELECT_OPTIONS}
          />
        }
      />
      {error ? <ErrorBanner message={error} onRetry={retry} /> : null}
      {loading ? (
        <div className="mt-6 space-y-4">
          <ChartSkeleton />
          <TableSkeleton />
        </div>
      ) : null}
      {!loading && globalExp === null ? (
        <div className="mt-6">
          <EmptyState
            title={`No global mean |SHAP| artifact for ${modelLabel(model)}`}
            body="Run `python -m ml.explain` so JSON exists under ARTIFACTS_DIR/explanations/. The API does not compute KernelSHAP on request, and this page does not fabricate feature contributions."
          />
        </div>
      ) : null}
      {globalExp ? (
        <div className="mt-6 space-y-6">
          <p className="font-mono text-xs text-mute">
            {modelLabel(globalExp.model_name)} · {globalExp.explainer}
            {globalExp.explainer_mode ? ` / ${globalExp.explainer_mode}` : ""}
            {globalExp.data_label ? ` · ${globalExp.data_label}` : ""}
            {globalExp.data_origin ? ` · origin ${globalExp.data_origin}` : ""} · n={globalExp.n_samples}
          </p>
          <OccupancyCaption source={occupancySource} />
          {globalExp.narrative ? (
            <p className="max-w-3xl text-sm leading-6 text-paper">{globalExp.narrative}</p>
          ) : null}
          <ChartPanel
            height={Math.max(280, ranking.length * 28)}
            empty={ranking.length === 0 ? "Global artifact has an empty ranking list" : null}
            caption="Global Feature Importance = mean |SHAP| (average magnitude on the explained subset). This is not a signed contribution to one prediction. Occupancy features are labeled simulated unless a measured source exists."
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={ranking}
                layout="vertical"
                margin={{ top: 8, right: 16, left: 8, bottom: 8 }}
              >
                <CartesianGrid stroke={SERIES.grid} strokeDasharray="3 6" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#8aa0b0", fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={140}
                  tick={{ fill: "#f0e6d6", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip content={<DeskTooltip />} />
                <Bar dataKey="shap" name="mean |SHAP|" fill={SERIES.predicted} maxBarSize={14} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
          {localExp ? (
            <div>
              <h3 className="font-display text-lg font-bold">Local explanation (stored example)</h3>
              <p className="mt-1 text-sm text-mute">
                Stored example from the last explain CLI, not live What-if. Change inputs on the What-if
                page and click Predict to compute SHAP for that row.
              </p>
              <p className="mt-1 font-mono text-xs text-mute">
                pred {formatKwh(localExp.prediction)}
                {localExp.building_id ? ` · ${localExp.building_id}` : ""}
                {localExp.ts ? ` · ${formatTs(localExp.ts)}` : ""}
                {localExp.row_index !== null && localExp.row_index !== undefined
                  ? ` · row ${localExp.row_index}`
                  : " · row 0"}
              </p>
              {localExp.narrative ? (
                <p className="mt-2 max-w-3xl text-sm leading-6 text-paper">{localExp.narrative}</p>
              ) : null}
              <table className="desk-table mt-3">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>Contribution</th>
                  </tr>
                </thead>
                <tbody>
                  {[...localExp.features]
                    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                    .slice(0, 12)
                    .map((row) => (
                      <tr key={row.name}>
                        <td>{occupancyFeatureLabel(row.name, localExp.occupancy_source)}</td>
                        <td className="tabular-nums">{formatMetric(row.value)}</td>
                        <td className={`tabular-nums ${row.contribution >= 0 ? "text-phosphor" : "text-rust"}`}>
                          {row.contribution >= 0 ? "+" : ""}
                          {formatMetric(row.contribution)}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              title="No local explanation"
              body="Global importance loaded, but /api/explainability with row=0 returned 404. Local SHAP is not synthesized in the browser."
            />
          )}
        </div>
      ) : null}
    </section>
  );
}
