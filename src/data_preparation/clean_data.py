"""Cleaning utilities for Sorare scores."""

from __future__ import annotations

import pandas as pd


POSITION_MAP = {
    "Goalkeeper": "GK",
    "goalkeeper": "GK",
    "GK": "GK",
    "Defender": "DEF",
    "defender": "DEF",
    "DEF": "DEF",
    "Midfielder": "MID",
    "midfielder": "MID",
    "MID": "MID",
    "Forward": "FWD",
    "forward": "FWD",
    "FWD": "FWD",
}


ALIASES = {
    "player_id": ["player_id", "Player Slug", "player_slug"],
    "player_name": ["player_name", "Player"],
    "team": ["team", "Club", "Selected Club"],
    "team_slug": ["team_slug", "Club Slug", "Selected Club Slug"],
    "position": ["position", "Position"],
    "match_date": ["match_date", "Game Date"],
    "gameweek": ["gameweek", "Sorare GW"],
    "competition": ["competition", "Competition"],
    "sorare_score": ["sorare_score", "Score"],
    "decisive_score": ["decisive_score", "Decisive Score"],
    "all_around_score": ["all_around_score", "All Around Score"],
    "minutes_played": ["minutes_played", "Minutes Played"],
    "price_eur": ["price_eur", "Price EUR", "market_value"],
}


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known Sorare/notebook fields to canonical names."""
    rename_map = {}
    for canonical, candidates in ALIASES.items():
        found = _first_existing(df, candidates)
        if found and found != canonical:
            rename_map[found] = canonical
    return df.rename(columns=rename_map)


def clean_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Clean score data and return reports for duplicates/invalid/outliers."""
    cleaned = standardise_columns(df).copy()
    reports: dict[str, pd.DataFrame] = {}

    required = ["player_id", "match_date"]
    missing = [col for col in required if col not in cleaned.columns]
    if missing:
        raise ValueError(f"Missing required columns after standardisation: {missing}")

    cleaned["match_date"] = pd.to_datetime(cleaned["match_date"], utc=True, errors="coerce")
    if "position" in cleaned.columns:
        cleaned["position"] = cleaned["position"].replace(POSITION_MAP)

    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["player_id", "match_date"]).copy()
    print(f"Removed {before - len(cleaned)} rows missing player_id or match_date.")

    if "sorare_score" in cleaned.columns:
        cleaned["sorare_score"] = pd.to_numeric(cleaned["sorare_score"], errors="coerce")
        cleaned["score_missing"] = cleaned["sorare_score"].isna()
        cleaned["score_is_zero"] = cleaned["sorare_score"].eq(0)
        reports["invalid_scores"] = cleaned[
            cleaned["sorare_score"].notna()
            & ~cleaned["sorare_score"].between(0, 100)
        ].copy()
        q1 = cleaned["sorare_score"].quantile(0.25)
        q3 = cleaned["sorare_score"].quantile(0.75)
        iqr = q3 - q1
        reports["score_outliers"] = cleaned[
            (cleaned["sorare_score"] < q1 - 1.5 * iqr)
            | (cleaned["sorare_score"] > q3 + 1.5 * iqr)
        ].copy()

    if "minutes_played" in cleaned.columns:
        cleaned["minutes_played"] = pd.to_numeric(cleaned["minutes_played"], errors="coerce")
        reports["invalid_minutes"] = cleaned[
            cleaned["minutes_played"].notna()
            & ~cleaned["minutes_played"].between(0, 130)
        ].copy()

    dup_cols = [col for col in ["player_id", "match_date", "competition"] if col in cleaned.columns]
    if len(dup_cols) >= 2:
        reports["duplicates"] = cleaned[cleaned.duplicated(subset=dup_cols, keep=False)].copy()
        cleaned = cleaned.drop_duplicates(subset=dup_cols, keep="last")

    return cleaned, reports
