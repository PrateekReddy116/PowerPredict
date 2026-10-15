import type { ReactNode } from "react";

type Props = {
  label: string;
  value: string;
  hint?: string;
  badge?: ReactNode;
  loading?: boolean;
};

export function KpiTile({ label, value, hint, badge, loading }: Props) {
  return (
    <article className="border border-zinc-800 bg-zinc-950 px-4 py-3">
      <div className="flex items-start justify-between gap-2">
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-400">{label}</p>
        {badge}
      </div>
      {loading ? (
        <div className="skeleton mt-3 h-8 w-28" aria-hidden />
      ) : (
        <p className="mt-2 font-display text-2xl font-bold tabular-nums tracking-tight text-white">{value}</p>
      )}
      {hint ? <p className="mt-1 text-[11px] leading-4 text-zinc-400">{hint}</p> : null}
    </article>
  );
}
