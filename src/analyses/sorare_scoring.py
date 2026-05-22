"""Sorare Football scoring matrix helpers.

The constants in this module are based on Sorare's public Football scoring
help article and its scoring-matrix images:
https://sorare.com/pt/help/a/4402904001809/how-does-scoring-work-in-sorare-football

Only a subset of matrix events is available in this repository's local
Transfermarkt-enriched data. Calculated scores are therefore estimates, not a
replacement for official Sorare scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


SCORING_SOURCE_URL = "https://sorare.com/pt/help/a/4402904001809/how-does-scoring-work-in-sorare-football"
SUPPORTED_MATRIX_VERSIONS = {"current", "new"}
POSITIONS = ("Goalkeeper", "Defender", "Midfielder", "Forward")


@dataclass(frozen=True)
class DecisiveLevel:
    level: int
    points: float
    guaranteed: bool


DECISIVE_SCORE_LEVELS: dict[int, DecisiveLevel] = {
    -3: DecisiveLevel(level=-3, points=0, guaranteed=False),
    -2: DecisiveLevel(level=-2, points=5, guaranteed=False),
    -1: DecisiveLevel(level=-1, points=15, guaranteed=False),
    0: DecisiveLevel(level=0, points=35, guaranteed=False),
    1: DecisiveLevel(level=1, points=60, guaranteed=True),
    2: DecisiveLevel(level=2, points=70, guaranteed=True),
    3: DecisiveLevel(level=3, points=80, guaranteed=True),
    4: DecisiveLevel(level=4, points=90, guaranteed=True),
    5: DecisiveLevel(level=5, points=100, guaranteed=True),
}

POSITIVE_DECISIVE_IMPACTS = (
    "goal",
    "assist",
    "penalty_won",
    "clearance_off_line",
    "clean_sheet_goalkeeper",
    "penalty_save",
    "last_man_tackle",
)

NEGATIVE_DECISIVE_IMPACTS = (
    "red_card",
    "own_goal",
    "penalty_conceded",
    "error_leads_to_goal",
)


def _event(
    group: str,
    event: str,
    goalkeeper: tuple[float, float],
    defender: tuple[float, float],
    midfielder: tuple[float, float],
    forward: tuple[float, float],
) -> dict[str, object]:
    return {
        "group": group,
        "event": event,
        "Goalkeeper": {"current": goalkeeper[0], "new": goalkeeper[1]},
        "Defender": {"current": defender[0], "new": defender[1]},
        "Midfielder": {"current": midfielder[0], "new": midfielder[1]},
        "Forward": {"current": forward[0], "new": forward[1]},
    }


ALL_AROUND_MATRIX = [
    _event("general", "yellow_card", (-3, -3), (-3, -3), (-3, -3), (-3, -3)),
    _event("general", "fouls", (0, -1), (-1, -2), (-0.5, -1), (0, -0.5)),
    _event("general", "was_fouled", (0, 0), (0, 0), (0.5, 1), (1, 1)),
    _event("general", "error_lead_to_shot", (-5, -5), (-1, -5), (-1, -3), (-1, -3)),
    _event("general", "clean_sheet_60", (0, 0), (10, 10), (0, 0), (0, 0)),
    _event("defensive", "goals_conceded", (0, -3), (-2, -4), (-2, -2), (0, 0)),
    _event("defensive", "effective_clearance", (0, 0), (0.5, 0.5), (0, 0), (0, 0)),
    _event("defensive", "won_tackle", (0, 0), (3, 3), (3, 3), (0, 0)),
    _event("defensive", "blocked_cross", (0, 0), (1, 1), (1, 1), (0, 0)),
    _event("defensive", "outfielder_block", (0, 0), (2, 2), (1, 1), (0, 0)),
    _event("defensive", "double_double", (0, 0), (4, 4), (4, 4), (4, 4)),
    _event("defensive", "triple_double", (0, 0), (6, 6), (6, 6), (6, 6)),
    _event("defensive", "triple_triple", (0, 0), (12, 12), (12, 12), (12, 12)),
    _event("possession", "poss_lost_ctrl", (-0.3, -0.3), (-1, -0.6), (-0.5, -0.5), (-0.1, -0.1)),
    _event("possession", "poss_won", (0, 0), (0.5, 0.5), (0.5, 0.5), (0, 0)),
    _event("possession", "duel_lost", (0, 0), (-2, -2), (-0.5, -0.8), (-0.5, -1)),
    _event("possession", "duel_won", (0, 0), (1.5, 1.5), (0.5, 0.8), (0.5, 1)),
    _event("possession", "interception_won", (0, 0), (3, 3), (2, 3), (0, 3)),
    _event("passing", "big_chance_created", (3, 3), (3, 3), (3, 3), (3, 3)),
    _event("passing", "adjusted_total_att_assist", (2, 2), (2, 3), (2, 2), (2, 2)),
    _event("passing", "accurate_pass", (0, 0.1), (0.1, 0.08), (0.1, 0.1), (0.1, 0.1)),
    _event("passing", "successful_final_third_passes", (0, 0.5), (0.5, 0.4), (0.3, 0.3), (0.1, 0.1)),
    _event("passing", "accurate_long_balls", (0.2, 0.2), (0.5, 0.5), (0.5, 0.5), (0, 0)),
    _event("passing", "long_pass_own_to_opp_success", (0, 0), (0.5, 0.5), (0, 0), (0, 0)),
    _event("passing", "missed_pass", (-0.2, -0.2), (-0.2, -0.2), (-0.2, -0.3), (0, 0)),
    _event("attacking", "ontarget_scoring_att", (3, 3), (3, 3), (3, 3), (3, 3)),
    _event("attacking", "won_contest", (0, 0), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
    _event("attacking", "pen_area_entries", (0, 0), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
    _event("attacking", "penalty_kick_missed", (-5, -5), (-5, -5), (-5, -5), (-5, -5)),
    _event("attacking", "big_chance_missed", (-5, -5), (-5, -5), (-5, -5), (-5, -5)),
    _event("goalkeeping", "saves", (2, 2), (0, 0), (0, 0), (0, 0)),
    _event("goalkeeping", "saved_ibox", (1, 2), (0, 0), (0, 0), (0, 0)),
    _event("goalkeeping", "good_high_claim", (1.2, 1.5), (0, 0), (0, 0), (0, 0)),
    _event("goalkeeping", "punches", (1.2, 1.5), (0, 0), (0, 0), (0, 0)),
    _event("goalkeeping", "dive_save", (3, 3), (0, 0), (0, 0), (0, 0)),
    _event("goalkeeping", "dive_catch", (3.5, 3.5), (0, 0), (0, 0), (0, 0)),
    _event("goalkeeping", "cross_not_claimed", (-5, -3), (0, 0), (0, 0), (0, 0)),
    _event("goalkeeping", "six_second_violation", (-5, -5), (0, 0), (0, 0), (0, 0)),
    _event("goalkeeping", "gk_smother", (5, 5), (0, 0), (0, 0), (0, 0)),
    _event("goalkeeping", "accurate_keeper_sweeper", (5, 3), (0, 0), (0, 0), (0, 0)),
]


LOCAL_EVENT_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "goals": ("TM Goals",),
    "assists": ("TM Assists",),
    "yellow_cards": ("TM Yellow Cards", "Event Yellow Cards"),
    "red_cards": ("TM Red Cards", "Event Red Cards"),
    "minutes": ("TM Appearance Minutes", "TM Minutes Played"),
    "goals_against": ("Team Goals Against",),
}


def normalize_position(value: object) -> str:
    """Map project position values to the Sorare matrix position labels."""
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text.startswith("goal") or text in {"gk", "keeper"}:
        return "Goalkeeper"
    if text.startswith("def") or text in {"cb", "lb", "rb", "wingback"}:
        return "Defender"
    if text.startswith("mid") or text in {"dm", "cm", "am"}:
        return "Midfielder"
    if text.startswith("for") or text in {"fw", "st", "winger", "attacker"}:
        return "Forward"
    return "Midfielder"


def _coerce_number(value: object, default: float = 0.0) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iat[0]
    if pd.isna(number):
        return default
    return float(number)


def first_numeric_value(row: pd.Series, candidates: tuple[str, ...], default: float = 0.0) -> float:
    """Return the first non-null numeric value found in a row."""
    for col in candidates:
        if col in row and pd.notna(row[col]):
            return _coerce_number(row[col], default=default)
    return default


def decisive_score_from_level(level: int) -> DecisiveLevel:
    return DECISIVE_SCORE_LEVELS[max(-3, min(5, int(level)))]


def all_around_event_value(event: str, position: str, version: str = "current") -> float:
    """Return one event's matrix value for a Sorare position and version."""
    version = version if version in SUPPORTED_MATRIX_VERSIONS else "current"
    position = normalize_position(position)
    for row in ALL_AROUND_MATRIX:
        if row["event"] == event:
            return float(row[position][version])
    return 0.0


def scoring_matrix_dataframe(version: str = "current") -> pd.DataFrame:
    """Return the All-Around matrix in a dashboard/report friendly table."""
    version = version if version in SUPPORTED_MATRIX_VERSIONS else "current"
    rows = []
    for row in ALL_AROUND_MATRIX:
        rows.append(
            {
                "group": row["group"],
                "event": row["event"],
                "goalkeeper": row["Goalkeeper"][version],
                "defender": row["Defender"][version],
                "midfielder": row["Midfielder"][version],
                "forward": row["Forward"][version],
                "matrix_version": version,
            }
        )
    return pd.DataFrame(rows)


def decisive_score_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"level": level, "points": data.points, "guaranteed_minimum": data.guaranteed}
            for level, data in DECISIVE_SCORE_LEVELS.items()
        ]
    ).sort_values("level")


def estimate_row_score(row: pd.Series, position_col: str = "Position", score_col: str | None = "Score", version: str = "current") -> dict[str, float | int | str]:
    """Estimate Sorare score components from locally available event columns."""
    version = version if version in SUPPORTED_MATRIX_VERSIONS else "current"
    position = normalize_position(row.get(position_col))
    goals = first_numeric_value(row, LOCAL_EVENT_COLUMNS["goals"])
    assists = first_numeric_value(row, LOCAL_EVENT_COLUMNS["assists"])
    red_cards = first_numeric_value(row, LOCAL_EVENT_COLUMNS["red_cards"])
    yellow_cards = first_numeric_value(row, LOCAL_EVENT_COLUMNS["yellow_cards"])
    minutes = first_numeric_value(row, LOCAL_EVENT_COLUMNS["minutes"])
    goals_against = first_numeric_value(row, LOCAL_EVENT_COLUMNS["goals_against"], default=np.nan)

    clean_sheet_60 = 1.0 if pd.notna(goals_against) and goals_against == 0 and minutes >= 60 else 0.0
    goalkeeper_clean_sheet = 1.0 if position == "Goalkeeper" and clean_sheet_60 else 0.0

    positive_impacts = goals + assists + goalkeeper_clean_sheet
    negative_impacts = red_cards
    level = int(max(-3, min(5, positive_impacts - negative_impacts)))
    decisive = decisive_score_from_level(level)

    aas = 0.0
    coverage_events = 0
    if yellow_cards:
        aas += yellow_cards * all_around_event_value("yellow_card", position, version)
        coverage_events += 1
    if clean_sheet_60:
        aas += clean_sheet_60 * all_around_event_value("clean_sheet_60", position, version)
        coverage_events += 1
    if pd.notna(goals_against) and goals_against:
        aas += goals_against * all_around_event_value("goals_conceded", position, version)
        coverage_events += 1

    raw_score = decisive.points + aas
    if decisive.guaranteed and raw_score < decisive.points:
        raw_score = decisive.points
    estimated_score = float(np.clip(raw_score, 0, 100))
    actual_score = _coerce_number(row.get(score_col), default=np.nan) if score_col else np.nan
    gap = actual_score - estimated_score if pd.notna(actual_score) else np.nan

    return {
        "sorare_matrix_version": version,
        "sorare_matrix_position": position,
        "sorare_decisive_positive_count": float(positive_impacts),
        "sorare_decisive_negative_count": float(negative_impacts),
        "sorare_decisive_level": level,
        "sorare_decisive_score": float(decisive.points),
        "sorare_decisive_guaranteed_minimum": bool(decisive.guaranteed),
        "sorare_all_around_estimate": float(aas),
        "sorare_matrix_score_estimate": estimated_score,
        "sorare_matrix_score_gap": float(gap) if pd.notna(gap) else np.nan,
        "sorare_matrix_coverage_events": int(coverage_events),
        "sorare_estimated_clean_sheet_60": bool(clean_sheet_60),
    }


def add_sorare_scoring_features(
    df: pd.DataFrame,
    position_col: str = "Position",
    score_col: str | None = "Score",
    version: str = "current",
) -> pd.DataFrame:
    """Add row-level Sorare matrix estimates using available local columns."""
    if df.empty:
        return df.copy()
    estimates = df.apply(
        lambda row: estimate_row_score(row, position_col=position_col, score_col=score_col, version=version),
        axis=1,
        result_type="expand",
    )
    return pd.concat([df.reset_index(drop=True), estimates.reset_index(drop=True)], axis=1)


def card_score(player_score: float, bonus_pct: float = 0.0) -> float:
    """Calculate Sorare Card Score from Player Score and card bonus percent."""
    score = _coerce_number(player_score, default=0)
    bonus = _coerce_number(bonus_pct, default=0)
    return float(score + score * bonus / 100)
