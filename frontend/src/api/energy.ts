import { fetchEnergyHistoryPages, withQuery } from "./client";
import type { Reading } from "./types";
import type { DataOrigin } from "../origin";

export function loadEnergyHistory(
  token: string,
  buildingId: string,
  opts: { origin?: DataOrigin; signal?: AbortSignal; maxPages?: number } = {},
) {
  return fetchEnergyHistoryPages<Reading>(
    (from) =>
      withQuery("/api/energy/history", {
        building_id: buildingId,
        from,
        origin: opts.origin,
      }),
    token,
    { signal: opts.signal, maxPages: opts.maxPages },
  );
}
