import type { ReactNode } from "react";

type Props = {
  height?: number;
  empty?: string | null;
  children: ReactNode;
  caption?: string;
};

export function ChartPanel({ height = 280, empty, children, caption }: Props) {
  return (
    <div>
      <div className="relative border border-zinc-800 bg-zinc-950" style={{ height }}>
        {empty ? (
          <p className="flex h-full items-center justify-center px-4 text-center font-mono text-[11px] uppercase tracking-[0.16em] text-zinc-400">
            {empty}
          </p>
        ) : (
          children
        )}
      </div>
      {caption ? <p className="mt-2 text-[11px] leading-4 text-zinc-400">{caption}</p> : null}
    </div>
  );
}
