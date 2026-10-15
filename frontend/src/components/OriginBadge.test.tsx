import { render, screen } from "@testing-library/react";

import { ORIGIN_LABELS, type DataOrigin } from "../origin";
import { OriginBadge } from "./OriginBadge";

const ORIGINS: DataOrigin[] = ["historical", "simulated_iot", "real_iot"];

describe("OriginBadge", () => {
  it("renders a distinct label for each origin", () => {
    render(
      <>
        {ORIGINS.map((origin) => (
          <OriginBadge key={origin} origin={origin} />
        ))}
      </>,
    );
    const badges = screen.getAllByTestId("origin-badge");
    expect(badges).toHaveLength(3);
    expect(screen.getByText(ORIGIN_LABELS.historical)).toBeInTheDocument();
    expect(screen.getByText(ORIGIN_LABELS.simulated_iot)).toBeInTheDocument();
    expect(screen.getByText(ORIGIN_LABELS.real_iot)).toBeInTheDocument();
    const classNames = badges.map((node) => node.className);
    expect(new Set(classNames).size).toBe(3);
  });
});
