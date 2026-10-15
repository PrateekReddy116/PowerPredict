type Props = {
  title: string;
  body: string;
};

export function EmptyState({ title, body }: Props) {
  return (
    <div className="border border-dashed border-hair bg-console/60 px-5 py-8">
      <p className="font-display text-lg text-paper">{title}</p>
      <p className="mt-2 max-w-xl text-sm leading-6 text-mute">{body}</p>
    </div>
  );
}
