"""Shared helpers for the Sorare player analysis dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT_DIR / "outputs"


PLAYER_COLS = ["Player", "player_name_standardized", "Player Slug"]
TEAM_COLS = ["Selected Club", "team_standardized", "Team", "club"]
LEAGUE_COLS = ["competition_id", "leagues_played", "League", "league"]
POSITION_COLS = ["Position", "position"]
MINUTES_COLS = ["total_minutes", "minutes", "Minutes"]


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first candidate column that exists in a dataframe."""
    available = set(columns)
    return next((column for column in candidates if column in available), None)


def metric_options(df: pd.DataFrame) -> list[str]:
    """Score-like columns that make useful ranking targets."""
    preferred = [
        "avg_score",
        "position_adjusted_score",
        "minutes_adjusted_score",
        "score_per_90",
        "consistency_score",
        "expected_score",
        "score_residual",
        "avg_sorare_matrix_score_estimate",
        "avg_sorare_matrix_score_gap",
        "avg_sorare_all_around_estimate",
        "avg_sorare_decisive_score",
        "total_score",
        "median_score",
    ]
    return [col for col in preferred if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def numeric_options(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> list[str]:
    """Return numeric columns, excluding identifiers and optional columns."""
    exclude_set = set(exclude or [])
    return [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_set and not col.lower().endswith("_id")
    ]


def safe_round(value: object, digits: int = 1, default: str = "N/A") -> str:
    """Format a number for KPI cards without leaking NaN into the UI."""
    try:
        if pd.isna(value):
            return default
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return default


def make_warning_markdown(warnings: list[str]) -> str:
    """Render missing-file and data-quality warnings for Gradio."""
    if not warnings:
        return "All expected dashboard files were found."
    items = "\n".join(f"- {warning}" for warning in warnings)
    return f"### Data availability warnings\n{items}"


def infer_player_label(row: pd.Series) -> str:
    """Build a readable player label from whatever identity columns exist."""
    for col in PLAYER_COLS:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return "Unknown player"


def describe_feature(feature: str) -> str:
    """Short plain-English explanations for common engineered variables."""
    mapping = {
        "total_minutes": "Total playing time. It often captures trust, availability, and role security.",
        "appearances": "Number of recorded matches. More appearances usually stabilize the score estimate.",
        "availability_rate": "Share of available match time played. It rewards reliable selection and fitness.",
        "avg_minutes": "Average minutes per appearance. Starters usually separate from substitute profiles here.",
        "age": "Player age. It can proxy development stage, peak years, and decline risk.",
        "height_cm": "Player height in centimeters. It may matter differently by position.",
        "position_adjusted_height": "Height compared with positional norms, useful for physical-profile context.",
        "score_std": "Score volatility. High values indicate boom-bust performance patterns.",
        "transfermarkt_match_confidence": "Confidence in the Transfermarkt match, useful for data quality context.",
        "avg_sorare_decisive_level": "Average decisive impact level estimated from available local event columns.",
        "avg_sorare_decisive_score": "Average decisive score implied by goals, assists, red cards, and goalkeeper clean sheets available locally.",
        "avg_sorare_all_around_estimate": "Partial All-Around estimate from local yellow-card, clean-sheet, and goals-conceded fields.",
        "avg_sorare_matrix_score_estimate": "Partial player score estimate using the Sorare matrix and available local stats.",
        "avg_sorare_matrix_score_gap": "Actual average score minus the partial Sorare matrix estimate. Large gaps often mean missing all-around events or data differences.",
    }
    return mapping.get(feature, "Model-derived or dataset variable. Inspect it with correlations and player examples.")


def empty_df(message: str) -> pd.DataFrame:
    """Return a one-cell table for empty states."""
    return pd.DataFrame({"message": [message]})
