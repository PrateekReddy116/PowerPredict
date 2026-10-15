import { formatNumber } from "../lib/format";

type PayloadItem = {
  name?: string | number;
  value?: string | number | Array<string | number>;
  color?: string;
};

type Props = {
  active?: boolean;
  payload?: PayloadItem[];
  label?: string | number;
  valueSuffix?: string;
};

export function DeskTooltip({ active, payload, label, valueSuffix }: Props) {
  if (!active || !payload?.length) return null;
  return (
    <div className="border border-hair bg-hull px-3 py-2 text-xs text-paper">
      {label !== undefined && label !== null ? (
        <p className="font-mono text-[10px] text-mute">{String(label)}</p>
      ) : null}
      <ul className="mt-1 space-y-0.5">
        {payload.map((item, index) => {
          const raw = Array.isArray(item.value) ? item.value[0] : item.value;
          const numeric = typeof raw === "number" ? formatNumber(raw, 2) : String(raw ?? "—");
          return (
            <li key={`${String(item.name)}-${index}`} className="flex justify-between gap-4">
              <span className="text-mute">{String(item.name ?? "")}</span>
              <span className="tabular-nums">
                {numeric}
                {valueSuffix ? ` ${valueSuffix}` : ""}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
