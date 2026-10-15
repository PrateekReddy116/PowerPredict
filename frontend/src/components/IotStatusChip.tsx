type IotFlag = "LIVE" | "SIMULATED" | "OFFLINE";

const STYLES: Record<IotFlag, string> = {
  OFFLINE:
    "border border-dashed border-mute/60 bg-transparent font-mono text-[10px] uppercase tracking-[0.16em] text-mute",
  SIMULATED:
    "border border-dashed border-sim bg-transparent font-mono text-[10px] uppercase tracking-[0.16em] text-sim",
  LIVE: "rounded-full border border-phosphor bg-phosphor font-display text-[11px] font-bold tracking-tight text-hull",
};

const HINTS: Record<IotFlag, string> = {
  OFFLINE: "MQTT broker unreachable. Historical queries still work.",
  SIMULATED: "Demo broker path / simulator — never labeled live.",
  LIVE: "Reserved for real_iot meter traffic.",
};

type Props = {
  status: IotFlag;
};

export function IotStatusChip({ status }: Props) {
  return (
    <span title={HINTS[status]} className={`inline-flex items-center px-2.5 py-0.5 ${STYLES[status]}`}>
      {status}
    </span>
  );
}
