export function isSimulatedOccupancy(source: string | null | undefined): boolean {
  return !source || source === "simulated";
}

export function occupancyCaption(source: string | null | undefined): string {
  if (isSimulatedOccupancy(source)) {
    return "Simulated occupancy profile — not a measured sensor";
  }
  return `Occupancy source: ${source}`;
}

export function occupancyFeatureLabel(name: string, source: string | null | undefined): string {
  if (!/occupancy/i.test(name)) return name;
  if (isSimulatedOccupancy(source)) return `${name} (simulated)`;
  return name;
}
