import type { DataOrigin } from "../origin";
import { ORIGIN_HINTS, ORIGIN_LABELS } from "../origin";

const STYLES: Record<DataOrigin, string> = {
  historical:
    "origin-historical border-l-[6px] border-archive bg-archive/15 text-archive font-sans tracking-wide",
  simulated_iot:
    "origin-simulated border border-dashed border-sim bg-transparent text-sim font-mono uppercase tracking-[0.14em]",
  real_iot:
    "origin-real rounded-full border border-phosphor bg-phosphor text-hull font-display font-bold tracking-tight",
};

type Props = {
  origin: DataOrigin;
};

export function OriginBadge({ origin }: Props) {
  return (
    <span
      data-testid="origin-badge"
      data-origin={origin}
      title={ORIGIN_HINTS[origin]}
      className={`inline-flex items-center px-2.5 py-0.5 text-[11px] leading-5 ${STYLES[origin]}`}
    >
      {ORIGIN_LABELS[origin]}
    </span>
  );
}
