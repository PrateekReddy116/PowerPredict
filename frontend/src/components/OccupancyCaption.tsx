import { occupancyCaption } from "../lib/occupancy";

type Props = {
  source: string | null | undefined;
};

export function OccupancyCaption({ source }: Props) {
  return <p className="text-sm text-sim">{occupancyCaption(source)}</p>;
}
