"""Baseline next-score prediction model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_baseline_model(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Predict next Sorare score using historical features and time split."""
    data = df.sort_values(["player_id", "match_date"]).copy()
    data["next_score"] = data.groupby("player_id")["sorare_score"].shift(-1)
    data = data.dropna(subset=["next_score"]).copy()

    feature_cols = [
        col for col in [
            "rolling_5_avg_score",
            "rolling_15_avg_score",
            "median_score",
            "score_volatility",
            "score_floor",
            "score_ceiling",
            "consistency_index",
            "form_delta",
            "dnp_rate",
            "minutes_played_rate",
            "score_per_eur",
        ]
        if col in data.columns
    ]
    if not feature_cols or len(data) < 20:
        empty = pd.DataFrame()
        return empty, {"limitations": "Not enough rows/features to train a reliable baseline model."}

    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    split_date = data["match_date"].quantile(0.8)
    train = data[data["match_date"] <= split_date]
    test = data[data["match_date"] > split_date]
    if train.empty or test.empty:
        return pd.DataFrame(), {"limitations": "Time split produced empty train or test set."}

    model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=3)
    model.fit(train[feature_cols], train["next_score"])
    preds = model.predict(test[feature_cols])
    results = test[["player_id", "match_date", "sorare_score", "next_score"]].copy()
    results["predicted_next_score"] = preds

    metrics = {
        "MAE": mean_absolute_error(test["next_score"], preds),
        "RMSE": mean_squared_error(test["next_score"], preds, squared=False),
        "R2": r2_score(test["next_score"], preds),
        "feature_importance": dict(zip(feature_cols, model.feature_importances_)),
        "limitations": "Baseline model only uses historical score features and does not include injuries, lineups, opponent, home/away, or transfer context.",
    }
    return results, metrics
