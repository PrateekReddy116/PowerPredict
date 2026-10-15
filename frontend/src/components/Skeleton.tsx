type Props = {
  className?: string;
};

export function Skeleton({ className = "h-4 w-full" }: Props) {
  return <div className={`skeleton ${className}`} aria-hidden />;
}

export function ChartSkeleton({ height = 280 }: { height?: number }) {
  return (
    <div className="border border-hair bg-console/40 p-4" style={{ height }} aria-busy="true">
      <Skeleton className="h-3 w-32" />
      <div className="mt-6 flex h-[calc(100%-2rem)] items-end gap-2">
        {Array.from({ length: 12 }).map((_, i) => (
          <Skeleton key={i} className="flex-1" />
        ))}
      </div>
    </div>
  );
}

export function TableSkeleton({ rows = 6 }: { rows?: number }) {
  return (
    <div className="space-y-2" aria-busy="true">
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-8 w-full" />
      ))}
    </div>
  );
}
