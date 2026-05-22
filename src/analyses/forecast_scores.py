r"""Train leakage-aware Sorare score forecasting models.

Run from the project root:

    C:\ProgramData\Anaconda3\python.exe src\analyses\forecast_scores.py

Outputs are written to analyses/score_variable_impact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
INPUT_FILE = PROJECT_ROOT / "data_preparation" / "datasets" / "combined" / "sorare_transfermarkt_context.csv"
OUTPUT_DIR = PROJECT_ROOT / "analyses" / "score_variable_impact"
TARGET = "Score"
DATE_COL = "Game Date"
PLAYER_COL = "Player Slug"
RANDOM_STATE = 42


LEAKAGE_COLUMNS = {
    TARGET,
    "TM Goals",
    "TM Assists",
    "TM Minutes Played",
    "TM Appearance Minutes",
    "Had Goal",
    "Had Assist",
    "Had Goal Contribution",
    "Team Goals For",
    "Team Goals Against",
    "Result",
    "Team Points",
    "TM Yellow Cards",
    "TM Red Cards",
    "Event Yellow Cards",
    "Event Red Cards",
    "Event Substitutions",
}


IDENTIFIER_COLUMNS = {
    "Player",
    "Player Slug",
    "sorare_player_norm",
    "tm_player_id",
    "tm_player_name",
    "Club Slug",
    "Selected Club Slug",
    "Matched Club Slug",
    "sorare_club_slug",
    "game_id",
    "tm_club_id",
    "club_id",
    "opponent_id",
    "competition_id",
    "Sorare GW API",
    "Sorare GW",
    "Matched Match Date",
    "Matched Date Diff Days",
    "match_date",
    "date",
}


PRE_MATCH_CONTEXT_FEATURES = [
    "Club",
    "Selected Club",
    "Position",
    "Score Scope",
    "Matched Club",
    "Matched In Dataset Club",
    "Opponent Club",
    "Home Away",
    "Team Rest Days",
    "Team Position",
    "Opponent Position",
    "Team Formation",
    "opponent_ppg",
    "opponent_goals_for_pg",
    "opponent_goals_against_pg",
    "opponent_win_rate",
    "Fixture Difficulty Score",
    "Fixture Difficulty",
    "Fixture Difficulty Label",
    "Context Match Available",
]


LINEUP_FEATURES = ["Lineup Status"]


HISTORY_FEATURES = [
    "player_games_before",
    "player_avg_score_before",
    "player_median_score_before",
    "player_std_score_before",
    "player_last_score",
    "player_rolling_3_score",
    "player_rolling_5_score",
    "player_rolling_10_score",
    "player_zero_rate_before",
    "team_avg_score_before",
    "team_games_before",
    "position_avg_score_before",
]


def load_dataset() -> pd.DataFrame:
    """Load and type the source dataset."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[TARGET, DATE_COL, PLAYER_COL]).copy()

    for col in df.columns:
        if df[col].dtype == "object":
            lowered = df[col].astype(str).str.strip().str.lower()
            values = lowered.dropna().unique().tolist()
            if values and set(values).issubset({"true", "false", "1", "0", "yes", "no"}):
                df[col] = lowered.isin(["true", "1", "yes"])

    return df.sort_values([PLAYER_COL, DATE_COL]).reset_index(drop=True)


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create player/team historical features without using the current score."""
    out = df.sort_values([PLAYER_COL, DATE_COL]).copy()
    player_group = out.groupby(PLAYER_COL, group_keys=False)[TARGET]
    shifted = player_group.shift(1)

    out["player_games_before"] = out.groupby(PLAYER_COL).cumcount()
    out["player_avg_score_before"] = shifted.groupby(out[PLAYER_COL]).expanding().mean().reset_index(level=0, drop=True)
    out["player_median_score_before"] = shifted.groupby(out[PLAYER_COL]).expanding().median().reset_index(level=0, drop=True)
    out["player_std_score_before"] = shifted.groupby(out[PLAYER_COL]).expanding().std().reset_index(level=0, drop=True)
    out["player_last_score"] = shifted
    out["player_rolling_3_score"] = shifted.groupby(out[PLAYER_COL]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    out["player_rolling_5_score"] = shifted.groupby(out[PLAYER_COL]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    out["player_rolling_10_score"] = shifted.groupby(out[PLAYER_COL]).rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    out["player_zero_rate_before"] = (
        shifted.fillna(0)
        .eq(0)
        .groupby(out[PLAYER_COL])
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    team_col = "Selected Club" if "Selected Club" in out.columns else "Club"
    team_shifted = out.groupby(team_col, group_keys=False)[TARGET].shift(1)
    out["team_games_before"] = out.groupby(team_col).cumcount()
    out["team_avg_score_before"] = team_shifted.groupby(out[team_col]).expanding().mean().reset_index(level=0, drop=True)

    if "Position" in out.columns:
        position_shifted = out.groupby("Position", group_keys=False)[TARGET].shift(1)
        out["position_avg_score_before"] = (
            position_shifted.groupby(out["Position"]).expanding().mean().reset_index(level=0, drop=True)
        )
    else:
        out["position_avg_score_before"] = np.nan

    return out


def choose_features(df: pd.DataFrame, include_lineup: bool) -> list[str]:
    """Return the feature list for a pre-match or lineup-aware model."""
    features = HISTORY_FEATURES + [col for col in PRE_MATCH_CONTEXT_FEATURES if col in df.columns]
    if include_lineup:
        features.extend([col for col in LINEUP_FEATURES if col in df.columns])

    excluded = LEAKAGE_COLUMNS | IDENTIFIER_COLUMNS | {DATE_COL}
    return [col for col in features if col in df.columns and col not in excluded]


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """Split model inputs into numeric and categorical columns."""
    numeric = [col for col in features if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])]
    categorical = [col for col in features if col not in numeric]
    return numeric, categorical


def make_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Build a random forest model for mixed numeric/categorical features."""
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=40)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=25,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the newest 20 percent of match dates as the test period."""
    unique_dates = pd.Series(df[DATE_COL].sort_values().unique())
    split_index = max(1, int(len(unique_dates) * 0.8))
    split_date = unique_dates.iloc[split_index]
    train = df[df[DATE_COL] < split_date].copy()
    test = df[df[DATE_COL] >= split_date].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty train or test set.")
    return train, test


def score_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Calculate forecast metrics."""
    mse = mean_squared_error(y_true, predictions)
    return {
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, predictions),
    }


def build_feature_importance(model: Pipeline, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    """Group random forest encoded feature importances back to original variables."""
    preprocessor: ColumnTransformer = model.named_steps["preprocess"]
    importances = model.named_steps["model"].feature_importances_

    names: list[str] = []
    names.extend(numeric_features)
    if categorical_features:
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        names.extend(onehot.get_feature_names_out(categorical_features).tolist())

    rows = []
    for encoded_name, importance in zip(names, importances):
        variable = encoded_name
        for cat_col in categorical_features:
            if encoded_name.startswith(f"{cat_col}_"):
                variable = cat_col
                break
        rows.append({"variable": variable, "encoded_feature": encoded_name, "importance": importance})

    return (
        pd.DataFrame(rows)
        .groupby("variable", dropna=False)
        .agg(total_importance=("importance", "sum"), max_encoded_importance=("importance", "max"))
        .reset_index()
        .sort_values("total_importance", ascending=False)
    )


def build_prediction_buckets(predictions: pd.DataFrame) -> pd.DataFrame:
    """Show how actual score changes across predicted-score buckets."""
    out = predictions.copy()
    out["predicted_bucket"] = pd.qcut(out["predicted_score"], q=5, duplicates="drop")
    return (
        out.groupby("predicted_bucket", observed=True)
        .agg(
            rows=("actual_score", "count"),
            avg_predicted=("predicted_score", "mean"),
            avg_actual=("actual_score", "mean"),
            median_actual=("actual_score", "median"),
            hit_50_plus=("actual_score", lambda s: float((s >= 50).mean())),
            zero_rate=("actual_score", lambda s: float((s == 0).mean())),
        )
        .reset_index()
    )


def train_one_model(df: pd.DataFrame, include_lineup: bool, name: str) -> dict[str, object]:
    """Train one forecast model and save its outputs."""
    features = choose_features(df, include_lineup=include_lineup)
    numeric_features, categorical_features = split_feature_types(df, features)
    train, test = temporal_split(df)

    train = train[train["player_games_before"] >= 2].copy()
    test = test[test["player_games_before"] >= 2].copy()
    model = make_model(numeric_features, categorical_features)
    model.fit(train[features], train[TARGET])

    predictions = model.predict(test[features])
    player_average_baseline = test["player_avg_score_before"].fillna(train[TARGET].mean()).to_numpy()
    global_baseline = np.full(len(test), train[TARGET].mean())

    metrics = score_predictions(test[TARGET], predictions)
    player_baseline_metrics = score_predictions(test[TARGET], player_average_baseline)
    global_baseline_metrics = score_predictions(test[TARGET], global_baseline)

    prediction_table = test[
        [
            DATE_COL,
            "Club",
            "Player",
            PLAYER_COL,
            "Position",
            "Opponent Club",
            "Home Away",
            "Fixture Difficulty Label",
            TARGET,
        ]
    ].copy()
    prediction_table = prediction_table.rename(columns={TARGET: "actual_score"})
    prediction_table["predicted_score"] = predictions
    prediction_table["prediction_error"] = prediction_table["actual_score"] - prediction_table["predicted_score"]
    prediction_table.sort_values("predicted_score", ascending=False).to_csv(
        OUTPUT_DIR / f"{name}_test_predictions.csv",
        index=False,
    )

    importance = build_feature_importance(model, numeric_features, categorical_features)
    importance.to_csv(OUTPUT_DIR / f"{name}_feature_importance.csv", index=False)

    buckets = build_prediction_buckets(prediction_table)
    buckets.to_csv(OUTPUT_DIR / f"{name}_prediction_buckets.csv", index=False)

    return {
        "name": name,
        "features": features,
        "train_rows": len(train),
        "test_rows": len(test),
        "metrics": metrics,
        "player_baseline_metrics": player_baseline_metrics,
        "global_baseline_metrics": global_baseline_metrics,
        "importance": importance,
        "buckets": buckets,
    }


def write_forecast_summary(results: list[dict[str, object]]) -> None:
    """Write a readable summary comparing forecast models."""
    lines = [
        "# Forecast Model Summary",
        "",
        "This uses a temporal split: older matches train the model, newer matches test it.",
        "",
    ]

    for result in results:
        metrics = result["metrics"]
        player_base = result["player_baseline_metrics"]
        global_base = result["global_baseline_metrics"]
        lines.extend(
            [
                f"## {result['name'].replace('_', ' ').title()}",
                "",
                f"- Train rows: {result['train_rows']:,}",
                f"- Test rows: {result['test_rows']:,}",
                f"- Model MAE: {metrics['mae']:.2f}",
                f"- Model RMSE: {metrics['rmse']:.2f}",
                f"- Model R2: {metrics['r2']:.3f}",
                f"- Player-average baseline MAE: {player_base['mae']:.2f}",
                f"- Global-average baseline MAE: {global_base['mae']:.2f}",
                "",
                "Top forecast features:",
                "",
            ]
        )
        importance = result["importance"]
        for _, row in importance.head(12).iterrows():
            lines.append(f"- {row['variable']}: {row['total_importance']:.4f}")
        lines.append("")

    lines.extend(
        [
            "## Reliability Notes",
            "",
            "- The pre-match model is the fair lineup-building model because it excludes match outcomes, minutes, goals, assists, result, and cards.",
            "- The lineup-aware model can be used only when you already know expected or official starter status.",
            "- Sorare scores are noisy, so use predictions as ranking signals, not exact point forecasts.",
            "- The model should be retrained whenever the dataset is refreshed.",
        ]
    )
    (OUTPUT_DIR / "forecast_model_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = add_history_features(load_dataset())

    results = [
        train_one_model(df, include_lineup=False, name="pre_match_forecast"),
        train_one_model(df, include_lineup=True, name="lineup_aware_forecast"),
    ]
    write_forecast_summary(results)

    for result in results:
        metrics = result["metrics"]
        player_base = result["player_baseline_metrics"]
        print(f"\n{result['name']}")
        print(f"Train rows: {result['train_rows']:,}")
        print(f"Test rows: {result['test_rows']:,}")
        print(f"Model MAE: {metrics['mae']:.2f}")
        print(f"Model RMSE: {metrics['rmse']:.2f}")
        print(f"Model R2: {metrics['r2']:.3f}")
        print(f"Player-average baseline MAE: {player_base['mae']:.2f}")
        print("Top features:")
        print(result["importance"].head(10).to_string(index=False))

    print(f"\nSaved forecast outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
