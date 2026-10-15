import type { ReactNode } from "react";

type Props = {
  kicker?: string;
  title: string;
  description: string;
  actions?: ReactNode;
};

export function PageHeader({ kicker, title, description, actions }: Props) {
  return (
    <div className="flex flex-col gap-4 border-b border-hair pb-5 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        {kicker ? (
          <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-copper">{kicker}</p>
        ) : null}
        <h2 className="font-display text-2xl font-bold tracking-tight text-white">{title}</h2>
        <p className="mt-1 max-w-2xl text-sm leading-6 text-zinc-300">{description}</p>
      </div>
      {actions ? <div className="flex shrink-0 flex-wrap items-center gap-3">{actions}</div> : null}
    </div>
  );
}
