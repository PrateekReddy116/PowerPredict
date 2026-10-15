import { aggregateEnergy, temperatureVsEnergy, type EnergyPoint } from "./aggregate";
import { formatOccupancy } from "./format";

const SAMPLE: EnergyPoint[] = [
  { ts: "2017-01-02T00:00:00Z", energy_kwh: 10, data_origin: "historical", temperature_c: 5 },
  { ts: "2017-01-02T01:00:00Z", energy_kwh: 12, data_origin: "historical", temperature_c: 6 },
  { ts: "2017-01-03T00:00:00Z", energy_kwh: 8, data_origin: "historical", temperature_c: null },
  { ts: "2017-01-09T00:00:00Z", energy_kwh: 20, data_origin: "simulated_iot", temperature_c: 2 },
];

describe("aggregateEnergy", () => {
  it("sums hourly points into daily totals", () => {
    const daily = aggregateEnergy(SAMPLE, "daily");
    expect(daily).toHaveLength(3);
    const first = daily.find((row) => row.key === "2017-01-02");
    expect(first?.kwh).toBe(22);
    expect(first?.n).toBe(2);
    expect(first?.origin).toBe("historical");
  });

  it("labels mixed origins when a bucket contains more than one", () => {
    const mixed: EnergyPoint[] = [
      { ts: "2017-01-02T00:00:00Z", energy_kwh: 1, data_origin: "historical" },
      { ts: "2017-01-02T01:00:00Z", energy_kwh: 1, data_origin: "simulated_iot" },
    ];
    const daily = aggregateEnergy(mixed, "daily");
    expect(daily[0]?.origin).toBe("mixed");
  });

  it("drops points without temperature from the scatter", () => {
    const scatter = temperatureVsEnergy(SAMPLE);
    expect(scatter).toHaveLength(3);
    expect(scatter.every((row) => typeof row.temperature_c === "number")).toBe(true);
  });
});

describe("formatOccupancy", () => {
  it("renders 0–1 feature occupancy and 0–100 MQTT occupancy as percent", () => {
    expect(formatOccupancy(0.9)).toBe("90% of typical capacity");
    expect(formatOccupancy(82)).toBe("82% of typical capacity");
    expect(formatOccupancy(null)).toBe("—");
  });
});
