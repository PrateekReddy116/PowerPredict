import { NavLink, Outlet } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";
import { OriginLegend } from "./OriginLegend";

const LINKS = [
  { to: "/overview", label: "Overview" },
  { to: "/forecast", label: "Forecasting" },
  { to: "/predict", label: "What-if" },
  { to: "/models", label: "Models" },
  { to: "/explain", label: "Explain" },
  { to: "/anomalies", label: "Anomalies" },
  { to: "/iot", label: "IoT" },
  { to: "/buildings", label: "Historical" },
];

export function AppShell() {
  const { email, role, logout } = useAuth();

  return (
    <div className="flex min-h-screen flex-col bg-black text-paper">
      <header className="border-b border-hair bg-black">
        <div className="flex flex-wrap items-end justify-between gap-4 px-5 pt-5 pb-4">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-copper">
              Energy forecasting desk
            </p>
            <h1 className="font-display text-3xl font-extrabold tracking-tight text-white">
              PowerPredict
            </h1>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <div className="text-right">
              <p className="font-mono text-xs text-zinc-300">{email}</p>
              <p className="font-display text-xs uppercase tracking-[0.16em] text-copper">{role}</p>
            </div>
            <button
              type="button"
              onClick={logout}
              className="border border-zinc-700 px-3 py-1.5 text-xs uppercase tracking-[0.14em] text-zinc-100 hover:border-copper hover:text-white"
            >
              Sign out
            </button>
          </div>
        </div>
        <div className="bus-bar" />
        <nav className="flex gap-1 overflow-x-auto bg-zinc-950 px-3 py-2" aria-label="Primary">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/overview"}
              className={({ isActive }) =>
                `shrink-0 border-b-2 px-3 py-2 font-display text-sm ${
                  isActive
                    ? "border-copper text-copper"
                    : "border-transparent text-zinc-400 hover:text-white"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <div className="flex flex-1 flex-col bg-black md:flex-row">
        <OriginLegend />
        <main className="min-w-0 flex-1 px-4 py-6 text-paper sm:px-5">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
