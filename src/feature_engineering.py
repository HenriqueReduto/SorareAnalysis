"""Feature engineering for Sorare scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling and aggregate player features without lookahead bias."""
    required = {"player_id", "match_date", "sorare_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {sorted(missing)}")

    out = df.sort_values(["player_id", "match_date"]).copy()
    grouped = out.groupby("player_id", group_keys=False)
    shifted = grouped["sorare_score"].shift(1)

    out["rolling_5_avg_score"] = shifted.groupby(out["player_id"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    out["rolling_15_avg_score"] = shifted.groupby(out["player_id"]).rolling(15, min_periods=1).mean().reset_index(level=0, drop=True)
    out["form_delta"] = out["rolling_5_avg_score"] - out["rolling_15_avg_score"]

    player_stats = grouped["sorare_score"].agg(
        median_score="median",
        score_volatility="std",
        score_floor="min",
        score_ceiling="max",
        avg_score="mean",
        games="count",
    )
    player_stats["consistency_index"] = player_stats["avg_score"] / (1 + player_stats["score_volatility"].fillna(0))
    out = out.merge(player_stats.reset_index(), on="player_id", how="left")

    if "minutes_played" in out.columns:
        out["played_flag"] = out["minutes_played"].fillna(0).gt(0)
        minutes_stats = out.groupby("player_id").agg(
            dnp_rate=("played_flag", lambda s: 1 - s.mean()),
            minutes_played_rate=("minutes_played", lambda s: s.fillna(0).clip(upper=90).mean() / 90),
        )
        out = out.merge(minutes_stats.reset_index(), on="player_id", how="left")
    else:
        out["dnp_rate"] = np.nan
        out["minutes_played_rate"] = np.nan

    value_col = "price_eur" if "price_eur" in out.columns else None
    if value_col:
        out["score_per_eur"] = out["avg_score"] / out[value_col].replace(0, np.nan)
    else:
        out["score_per_eur"] = np.nan

    return out
