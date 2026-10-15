import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { apiGet, apiPost, AuthError, type LoginResponse, type MeResponse } from "../api/client";

const TOKEN_KEY = "pp_access_token";

type AuthState = {
  token: string | null;
  email: string | null;
  role: "viewer" | "admin" | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => sessionStorage.getItem(TOKEN_KEY));
  const [email, setEmail] = useState<string | null>(null);
  const [role, setRole] = useState<"viewer" | "admin" | null>(null);
  const [loading, setLoading] = useState(Boolean(token));

  const logout = useCallback(() => {
    sessionStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setEmail(null);
    setRole(null);
  }, []);

  useEffect(() => {
    if (!token) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    apiGet<MeResponse>("/api/v1/auth/me", token)
      .then((me) => {
        if (cancelled) return;
        setEmail(me.email);
        setRole(me.role);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof AuthError) {
          logout();
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [token, logout]);

  const login = useCallback(async (nextEmail: string, password: string) => {
    const body = await apiPost<LoginResponse>("/api/v1/auth/login", {
      email: nextEmail,
      password,
    });
    sessionStorage.setItem(TOKEN_KEY, body.access_token);
    setToken(body.access_token);
    setEmail(body.email);
    setRole(body.role);
  }, []);

  const value = useMemo(
    () => ({ token, email, role, loading, login, logout }),
    [token, email, role, loading, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return ctx;
}
