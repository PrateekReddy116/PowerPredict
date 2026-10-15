import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts";

const EMPTY: { ts: string; kwh: number }[] = [];

export function EmptyChart() {
  return (
    <div className="relative h-56 border border-hair bg-console/40">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={EMPTY} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
          <XAxis dataKey="ts" hide />
          <YAxis hide />
          <Line type="monotone" dataKey="kwh" stroke="#d4894a" dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <p className="pointer-events-none absolute inset-0 flex items-center justify-center font-mono text-xs uppercase tracking-[0.18em] text-mute">
        Chart frame — no series loaded
      </p>
    </div>
  );
}
