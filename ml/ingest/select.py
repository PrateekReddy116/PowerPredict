"""Select 2–3 Education/Office buildings with high electricity completeness."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.constants import MIN_COMPLETENESS, N_BUILDINGS, PREFERRED_PRIMARY_USES


@dataclass
class BuildingChoice:
    building_id: str
    site_id: str
    primary_use: str | None
    sqm: float | None
    timezone: str
    completeness: float
    selected_reason: str


def candidate_metadata(
    metadata: pd.DataFrame,
    *,
    primary_uses: tuple[str, ...] = PREFERRED_PRIMARY_USES,
) -> pd.DataFrame:
    uses = {item.lower() for item in primary_uses}
    work = metadata.copy()
    use_ok = work["primary_use"].fillna("").astype(str).str.lower().isin(uses)
    elec = work["electricity"].fillna("").astype(str).str.strip().str.lower()
    has_electricity = elec.eq("yes") | elec.eq("true") | elec.eq("1")
    return work.loc[use_ok & has_electricity].copy()


def select_buildings(
    metadata: pd.DataFrame,
    completeness: pd.Series,
    *,
    n: int = N_BUILDINGS,
    min_completeness: float = MIN_COMPLETENESS,
    primary_uses: tuple[str, ...] = PREFERRED_PRIMARY_USES,
) -> list[BuildingChoice]:
    candidates = candidate_metadata(metadata, primary_uses=primary_uses)
    if candidates.empty:
        raise ValueError("no Education/Office buildings with an electricity meter in metadata")

    comp = completeness.rename("completeness")
    merged = candidates.merge(comp, left_on="building_id", right_index=True, how="inner")
    if merged.empty:
        raise ValueError("metadata candidates were not present in the electricity file")

    eligible = merged.loc[merged["completeness"] >= min_completeness]
    pool = eligible if len(eligible) >= min(n, 2) else merged
    ranked = pool.sort_values(["completeness", "building_id"], ascending=[False, True])

    # Prefer spreading across at most two sites while keeping the highest-completeness rows.
    picked: list[pd.Series] = []
    seen_sites: list[str] = []
    for _, row in ranked.iterrows():
        if len(picked) >= n:
            break
        site = str(row["site_id"])
        if len(seen_sites) >= 2 and site not in seen_sites:
            continue
        picked.append(row)
        if site not in seen_sites:
            seen_sites.append(site)
    if len(picked) < min(n, 2):
        picked = [row for _, row in ranked.head(n).iterrows()]

    choices: list[BuildingChoice] = []
    for row in picked[:n]:
        completeness_pct = 100.0 * float(row["completeness"])
        use = None if pd.isna(row.get("primary_use")) else str(row["primary_use"])
        bid = str(row["building_id"])
        if bid.startswith("TEST_FIXTURE"):
            origin = "TEST FIXTURE, not BDG2 (synthetic). "
        else:
            origin = "BDG2 cleaned electricity; "
        reason = (
            f"{origin}{use or 'unknown'} at site {row['site_id']}; "
            f"hourly completeness {completeness_pct:.2f}% "
            f"(threshold {100.0 * min_completeness:.0f}%). "
            "Preferred Education/Office with high completeness; not the full 1636-building corpus."
        )
        sqm = None if pd.isna(row.get("sqm")) else float(row["sqm"])
        choices.append(
            BuildingChoice(
                building_id=str(row["building_id"]),
                site_id=str(row["site_id"]),
                primary_use=use,
                sqm=sqm,
                timezone=str(row["timezone"]),
                completeness=float(row["completeness"]),
                selected_reason=reason,
            )
        )
    if len(choices) < 2:
        raise ValueError("could not select at least 2 buildings")
    return choices


def choices_frame(choices: list[BuildingChoice]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": c.building_id,
                "site_id": c.site_id,
                "primary_use": c.primary_use,
                "sqm": c.sqm,
                "timezone": c.timezone,
                "selected_reason": c.selected_reason,
                "electricity_completeness": c.completeness,
            }
            for c in choices
        ]
    )
