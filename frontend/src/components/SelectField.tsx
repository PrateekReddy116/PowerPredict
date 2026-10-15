import type { ChangeEvent } from "react";

type Option = {
  value: string;
  label: string;
};

type Props = {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Option[];
  disabled?: boolean;
};

export function SelectField({ id, label, value, onChange, options, disabled }: Props) {
  function handle(event: ChangeEvent<HTMLSelectElement>) {
    onChange(event.target.value);
  }

  return (
    <label className="flex min-w-[10rem] flex-col gap-1 text-sm" htmlFor={id}>
      <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-zinc-400">{label}</span>
      <select
        id={id}
        value={value}
        onChange={handle}
        disabled={disabled || options.length === 0}
        className="border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-white disabled:opacity-50"
      >
        {options.length === 0 ? <option value="">None available</option> : null}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
