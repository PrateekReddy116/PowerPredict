---
inclusion: fileMatch
fileMatchPattern: ['frontend/**/*.{ts,tsx}']
---

# Frontend provenance

## Origin labels (required)

Every reading, series, and table row must show origin. Never style Simulated, Real, and Historical the same.

- `historical` — BDG2
- `simulated_iot` — MQTT simulator
- `real_iot` — real sensor path
- Occupancy: show `occupancy_source` (simulated unless real data exists)

## Metrics and forecasts

Render MAE / RMSE / R² / MAPE from the API (`model_runs`). Do not hardcode or invent scores. If the API has no run, show an empty/error state.

## Stack

React function components, TypeScript, Tailwind, Recharts. Auth-gate all routes except `/login`. Poll IoT every ~5s; do not add WebSockets.

```tsx
// BAD
<span>Live sensor</span>  // hides that it is simulated

// GOOD
<OriginBadge origin={reading.data_origin} />
```
