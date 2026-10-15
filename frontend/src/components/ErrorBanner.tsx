type Props = {
  message: string;
  onRetry?: () => void;
};

export function ErrorBanner({ message, onRetry }: Props) {
  return (
    <div
      className="mt-4 flex flex-wrap items-center justify-between gap-3 border border-rust/40 bg-rust/10 px-4 py-3"
      role="alert"
    >
      <p className="text-sm text-paper">{message}</p>
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="border border-hair px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.14em] text-paper hover:border-copper"
        >
          Retry
        </button>
      ) : null}
    </div>
  );
}
