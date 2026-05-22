"""Analysis tables for Sorare scores."""

from __future__ import annotations

import pandas as pd


def build_rankings(df: pd.DataFrame, min_games: int = 5) -> dict[str, pd.DataFrame]:
    """Build ranking tables from engineered data."""
    group_cols = ["player_id"]
    for col in ["player_name", "team", "position"]:
        if col in df.columns:
            group_cols.append(col)

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            games=("sorare_score", "count"),
            avg_score=("sorare_score", "mean"),
            median_score=("sorare_score", "median"),
            score_volatility=("sorare_score", "std"),
            score_floor=("sorare_score", "min"),
            score_ceiling=("sorare_score", "max"),
            recent_form=("rolling_5_avg_score", "last"),
            longer_form=("rolling_15_avg_score", "last"),
            form_delta=("form_delta", "last"),
            consistency_index=("consistency_index", "last"),
            score_per_eur=("score_per_eur", "last"),
        )
        .reset_index()
    )
    eligible = summary[summary["games"] >= min_games].copy()
    tables = {
        "player_rankings": summary.sort_values(["avg_score", "games"], ascending=[False, False]),
        "most_consistent_players": eligible.sort_values("consistency_index", ascending=False),
        "highest_average_scores": eligible.sort_values("avg_score", ascending=False),
        "highest_ceiling_players": eligible.sort_values("score_ceiling", ascending=False),
        "improving_players": eligible.sort_values("form_delta", ascending=False),
        "declining_players": eligible.sort_values("form_delta", ascending=True),
        "high_risk_high_reward_players": eligible.assign(
            risk_reward=lambda x: x["score_ceiling"] - x["score_floor"]
        ).sort_values("risk_reward", ascending=False),
    }
    if "score_per_eur" in eligible.columns and eligible["score_per_eur"].notna().any():
        tables["best_value_players"] = eligible.sort_values("score_per_eur", ascending=False)
    else:
        tables["best_value_players"] = eligible.head(0)

    if "position" in summary.columns:
        tables["rankings_by_position"] = summary.sort_values(["position", "avg_score"], ascending=[True, False])
    else:
        tables["rankings_by_position"] = summary.head(0)
    return tables


def write_summary(path: str, tables: dict[str, pd.DataFrame]) -> None:
    """Write a markdown summary with insight format."""
    lines = ["# Sorare Analysis Summary\n"]
    for name, table in tables.items():
        if table.empty:
            continue
        top = table.iloc[0]
        player = top.get("player_name", top.get("player_id", "Unknown player"))
        lines.extend([
            f"## {name.replace('_', ' ').title()}",
            f"- Finding: `{player}` leads this table.",
            f"- Evidence: Top row has average score `{top.get('avg_score', 'n/a')}` over `{top.get('games', 'n/a')}` games.",
            "- Fantasy/gameplay impact: Use this table to separate stable picks from upside or risk plays.",
            "- Recommended action: Review the top 10 with recent form and position context before lineup decisions.\n",
        ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
