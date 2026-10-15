import { type FormEvent, useState } from "react";
import { Navigate } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";
import { OriginBadge } from "../components/OriginBadge";

export function LoginPage() {
  const { token, login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (token) {
    return <Navigate to="/overview" replace />;
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(email, password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign in failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="relative min-h-screen bg-black">
      <div className="pointer-events-none absolute inset-y-0 left-[12%] hidden w-px bg-zinc-800 md:block" />
      <div className="pointer-events-none absolute inset-y-0 left-[12%] hidden w-3 bg-copper/15 md:block" />
      <div className="mx-auto flex min-h-screen max-w-lg flex-col justify-center px-6 py-16">
        <p className="font-mono text-[10px] uppercase tracking-[0.3em] text-copper">
          Facilities operations
        </p>
        <h1 className="mt-3 font-display text-4xl font-extrabold tracking-tight text-white">
          PowerPredict
        </h1>
        <p className="mt-3 max-w-md text-sm leading-6 text-zinc-300">
          Electricity forecasting desk for selected buildings. Every series is labeled by origin —
          BDG2 archive, MQTT simulator, or live meter. Simulated occupancy is never shown as a
          measured sensor.
        </p>
        <div className="mt-5 flex flex-wrap gap-2">
          <OriginBadge origin="historical" />
          <OriginBadge origin="simulated_iot" />
          <OriginBadge origin="real_iot" />
        </div>
        <div className="bus-bar mt-8" />
        <form onSubmit={onSubmit} className="mt-8 flex flex-col gap-4">
          <label className="flex flex-col gap-1.5 text-sm">
            <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-zinc-400">
              Email
            </span>
            <input
              type="email"
              autoComplete="username"
              required
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              className="border border-zinc-700 bg-zinc-950 px-3 py-2 text-white"
            />
          </label>
          <label className="flex flex-col gap-1.5 text-sm">
            <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-zinc-400">
              Password
            </span>
            <input
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="border border-zinc-700 bg-zinc-950 px-3 py-2 text-white"
            />
          </label>
          {error ? <p className="text-sm text-sim">{error}</p> : null}
          <button
            type="submit"
            disabled={submitting}
            className="mt-2 bg-copper px-4 py-2.5 font-display text-sm font-bold text-black disabled:opacity-60"
          >
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}
