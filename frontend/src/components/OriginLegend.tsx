import { OriginBadge } from "./OriginBadge";
import type { DataOrigin } from "../origin";
import { ORIGIN_HINTS } from "../origin";

const ORDER: DataOrigin[] = ["historical", "simulated_iot", "real_iot"];

export function OriginLegend() {
  return (
    <aside
      className="flex w-full shrink-0 flex-col gap-4 border-b border-zinc-800 bg-zinc-950 px-4 py-4 md:w-52 md:border-b-0 md:border-r md:py-6"
      aria-label="Data origin legend"
    >
      <p className="font-display text-[11px] font-bold uppercase tracking-[0.22em] text-zinc-400">
        Origin key
      </p>
      <ul className="flex flex-row flex-wrap gap-4 md:flex-col md:gap-5">
        {ORDER.map((origin) => (
          <li key={origin} className="flex min-w-[9rem] flex-col items-start gap-1.5">
            <OriginBadge origin={origin} />
            <span className="text-[11px] leading-4 text-zinc-400">{ORIGIN_HINTS[origin]}</span>
          </li>
        ))}
      </ul>
    </aside>
  );
}
