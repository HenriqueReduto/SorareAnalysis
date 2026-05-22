"""Reusable dataframe filters for dashboard callbacks."""

from __future__ import annotations

import re

import pandas as pd

from .helpers import LEAGUE_COLS, MINUTES_COLS, POSITION_COLS, TEAM_COLS, first_existing


def choices(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    col = first_existing(df.columns, candidates)
    if not col:
        return []
    values = df[col].dropna().astype(str)
    exploded = values.str.split(",").explode().str.strip()
    return sorted(v for v in exploded.unique().tolist() if v)


def age_bounds(df: pd.DataFrame) -> tuple[int, int]:
    if "age" not in df.columns or df["age"].dropna().empty:
        return 14, 50
    ages = pd.to_numeric(df["age"], errors="coerce").dropna()
    if ages.empty:
        return 14, 50
    return int(ages.min()), int(ages.max())


def apply_player_filters(
    df: pd.DataFrame,
    positions: list[str] | None = None,
    teams: list[str] | None = None,
    leagues: list[str] | None = None,
    age_range: tuple[int, int] | list[int] | None = None,
    min_minutes: float | int | None = None,
) -> pd.DataFrame:
    """Apply common player filters while tolerating absent columns."""
    if df.empty:
        return df

    out = df.copy()
    position_col = first_existing(out.columns, POSITION_COLS)
    team_col = first_existing(out.columns, TEAM_COLS)
    league_col = first_existing(out.columns, LEAGUE_COLS)
    minutes_col = first_existing(out.columns, MINUTES_COLS)

    if positions and position_col:
        out = out[out[position_col].astype(str).isin(positions)]
    if teams and team_col:
        out = out[out[team_col].astype(str).isin(teams)]
    if leagues and league_col:
        pattern = "|".join(rf"(^|,\s*){re.escape(str(x))}(\s*,|$)" for x in leagues)
        out = out[out[league_col].astype(str).str.contains(pattern, case=False, na=False, regex=True)]
    if age_range and "age" in out.columns:
        lo, hi = age_range
        out = out[out["age"].between(lo, hi, inclusive="both") | out["age"].isna()]
    if min_minutes is not None and minutes_col:
        out = out[pd.to_numeric(out[minutes_col], errors="coerce").fillna(0) >= float(min_minutes)]

    return out
