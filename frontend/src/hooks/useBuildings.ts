import { useCallback, useEffect, useState } from "react";

import { apiGet, AuthError } from "../api/client";
import type { Building } from "../api/types";
import { useAuth } from "../auth/AuthContext";

type State = {
  buildings: Building[];
  loading: boolean;
  error: string | null;
  retry: () => void;
};

export function useBuildings(): State {
  const { token, logout } = useAuth();
  const [buildings, setBuildings] = useState<Building[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  const retry = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!token) return;
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    apiGet<Building[]>("/api/buildings", token, { signal: ac.signal })
      .then((rows) => {
        setBuildings(rows);
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        if (err instanceof AuthError) logout();
        else setError(err instanceof Error ? err.message : "Could not load buildings");
      })
      .finally(() => {
        if (!ac.signal.aborted) setLoading(false);
      });
    return () => ac.abort();
  }, [token, logout, nonce]);

  return { buildings, loading, error, retry };
}
