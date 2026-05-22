"""Season-aware squad matching for player score rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SQUAD_ALIASES = {
    "club": ["club", "Club"],
    "club_slug": ["club_slug", "Club Slug"],
    "player_slug": ["player_slug", "Player Slug"],
    "season_start": ["season_start", "Season Start", "start_date"],
    "season_end": ["season_end", "Season End", "end_date"],
}


def _rename_squad_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for target, candidates in SQUAD_ALIASES.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = target
                break
    return df.rename(columns=rename)


def load_squad_memberships(path: str | Path) -> pd.DataFrame:
    """Load season squad memberships.

    Expected columns:
    - club or Club
    - club_slug or Club Slug
    - player_slug or Player Slug
    - season_start or Season Start
    - season_end or Season End
    """
    squad_path = Path(path)
    if not squad_path.exists():
        raise FileNotFoundError(f"Squad membership file not found: {squad_path}")

    squads = pd.read_csv(squad_path)
    squads = _rename_squad_columns(squads)
    required = {"club", "club_slug", "player_slug", "season_start", "season_end"}
    missing = required - set(squads.columns)
    if missing:
        raise ValueError(f"Squad file is missing required columns: {sorted(missing)}")

    squads["season_start"] = pd.to_datetime(squads["season_start"], utc=True, errors="coerce")
    squads["season_end"] = pd.to_datetime(squads["season_end"], utc=True, errors="coerce")
    squads = squads.dropna(subset=["player_slug", "season_start", "season_end"]).copy()
    return squads


def assign_club_by_season(
    scores: pd.DataFrame,
    squads: pd.DataFrame,
    player_col: str = "Player Slug",
    date_col: str = "Game Date",
) -> pd.DataFrame:
    """Assign each score to a tracked club only when the player was in that season squad.

    Rows that do not match any tracked club-season membership keep null matched club fields.
    """
    out = scores.copy()
    out[date_col] = pd.to_datetime(out[date_col], utc=True, errors="coerce")
    out["_score_row_id"] = range(len(out))

    lookup = squads.rename(columns={"player_slug": player_col}).copy()
    merged = out.merge(
        lookup[["club", "club_slug", player_col, "season_start", "season_end"]],
        on=player_col,
        how="left",
    )
    matched = merged[
        (merged[date_col] >= merged["season_start"])
        & (merged[date_col] <= merged["season_end"])
    ].copy()

    matched = matched.sort_values(["_score_row_id", "season_start"]).drop_duplicates("_score_row_id", keep="last")
    match_cols = matched[["_score_row_id", "club", "club_slug", "season_start", "season_end"]].rename(
        columns={
            "club": "Matched Club",
            "club_slug": "Matched Club Slug",
            "season_start": "Matched Season Start",
            "season_end": "Matched Season End",
        }
    )

    out = out.merge(match_cols, on="_score_row_id", how="left").drop(columns=["_score_row_id"])
    out["Matched In Dataset Club"] = out["Matched Club Slug"].notna()
    return out
