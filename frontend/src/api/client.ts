export class AuthError extends Error {
  constructor() {
    super("Not authenticated");
    this.name = "AuthError";
  }
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export type QueryValue = string | number | boolean | null | undefined;

export function withQuery(path: string, params: Record<string, QueryValue> = {}): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    search.set(key, String(value));
  }
  const qs = search.toString();
  return qs ? `${path}?${qs}` : path;
}

type RequestInitAuth = {
  signal?: AbortSignal;
};

async function parseJson<T>(response: Response): Promise<T> {
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

function messageForStatus(status: number, detail: unknown): string {
  if (status === 429) {
    return "Too many requests — wait and retry.";
  }
  if (typeof detail === "object" && detail && "detail" in detail) {
    const value = (detail as { detail: unknown }).detail;
    if (typeof value === "string") return value;
  }
  return `Request failed (${status})`;
}

async function rejectIfFailed(response: Response): Promise<void> {
  if (response.status === 401) {
    throw new AuthError();
  }
  if (response.ok) return;
  const detail = await response.json().catch(() => ({}));
  throw new ApiError(response.status, messageForStatus(response.status, detail));
}

export async function apiGet<T>(path: string, token: string, init: RequestInitAuth = {}): Promise<T> {
  const response = await fetch(path, {
    headers: { Authorization: `Bearer ${token}` },
    signal: init.signal,
  });
  await rejectIfFailed(response);
  return parseJson<T>(response);
}

export async function apiGetOptional<T>(
  path: string,
  token: string,
  init: RequestInitAuth = {},
): Promise<T | null> {
  const response = await fetch(path, {
    headers: { Authorization: `Bearer ${token}` },
    signal: init.signal,
  });
  if (response.status === 401) {
    throw new AuthError();
  }
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new ApiError(response.status, messageForStatus(response.status, detail));
  }
  return parseJson<T>(response);
}

export async function apiPost<T>(path: string, body: unknown, init: RequestInitAuth = {}): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: init.signal,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new ApiError(response.status, messageForStatus(response.status, detail));
  }
  return parseJson<T>(response);
}

export async function apiPostAuth<T>(
  path: string,
  body: unknown,
  token: string,
  init: RequestInitAuth = {},
): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: init.signal,
  });
  if (response.status === 401) throw new AuthError();
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new ApiError(response.status, messageForStatus(response.status, detail));
  }
  return parseJson<T>(response);
}

/** Like apiPostAuth, but returns null on 404 (e.g. model artifact not trained). */
export async function apiPostAuthOptional<T>(
  path: string,
  body: unknown,
  token: string,
  init: RequestInitAuth = {},
): Promise<T | null> {
  const response = await fetch(path, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: init.signal,
  });
  if (response.status === 401) throw new AuthError();
  if (response.status === 404) return null;
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new ApiError(response.status, messageForStatus(response.status, detail));
  }
  return parseJson<T>(response);
}

export type LoginResponse = {
  access_token: string;
  token_type: string;
  role: "viewer" | "admin";
  email: string;
};

export type MeResponse = {
  id: string;
  email: string;
  role: "viewer" | "admin";
  created_at: string;
};

/** History is oldest-first, capped at 10_000 rows. Page forward on `ts` to reach later hours. */
export const HISTORY_PAGE_SIZE = 10_000;

export async function fetchEnergyHistoryPages<T extends { ts: string; id?: string }>(
  pathBuilder: (fromTs: string | undefined) => string,
  token: string,
  init: RequestInitAuth & { maxPages?: number } = {},
): Promise<{ rows: T[]; truncated: boolean }> {
  const maxPages = init.maxPages ?? 3;
  const rows: T[] = [];
  const seen = new Set<string>();
  let fromTs: string | undefined;
  let truncated = false;

  for (let page = 0; page < maxPages; page += 1) {
    const batch = await apiGet<T[]>(pathBuilder(fromTs), token, { signal: init.signal });
    if (batch.length === 0) break;
    for (const row of batch) {
      const key = row.id ?? `${row.ts}:${rows.length}`;
      if (seen.has(key)) continue;
      seen.add(key);
      rows.push(row);
    }
    if (batch.length < HISTORY_PAGE_SIZE) {
      truncated = false;
      break;
    }
    fromTs = batch[batch.length - 1]?.ts;
    truncated = page === maxPages - 1;
    if (!fromTs) break;
  }

  return { rows, truncated };
}
