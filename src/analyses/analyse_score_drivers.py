r"""Analyze which context variables have the biggest impact on Sorare scores.

Run from the project root:

    C:\ProgramData\Anaconda3\python.exe src\analyses\analyse_score_drivers.py

The script writes CSV tables and a markdown summary to analyses/score_variable_impact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
INPUT_FILE = PROJECT_ROOT / "data_preparation" / "datasets" / "combined" / "sorare_transfermarkt_context.csv"
OUTPUT_DIR = PROJECT_ROOT / "analyses" / "score_variable_impact"
TARGET = "Score"
RANDOM_STATE = 42


EXCLUDE_COLUMNS = {
    TARGET,
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
    "Game Date",
    "Matched Match Date",
    "Matched Date Diff Days",
    "match_date",
    "date",
}


PREFERRED_FEATURES = [
    "Club",
    "Selected Club",
    "Position",
    "Score Scope",
    "Matched Club",
    "Lineup Status",
    "TM Goals",
    "TM Assists",
    "TM Minutes Played",
    "Matched In Dataset Club",
    "Had Goal",
    "Had Assist",
    "Had Goal Contribution",
    "Opponent Club",
    "Home Away",
    "Team Goals For",
    "Team Goals Against",
    "Result",
    "Team Points",
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
    "TM Yellow Cards",
    "TM Red Cards",
    "TM Appearance Minutes",
    "Event Yellow Cards",
    "Event Red Cards",
    "Event Substitutions",
    "Context Match Available",
]


def load_dataset(path: Path = INPUT_FILE) -> pd.DataFrame:
    """Load the context dataset and normalize common boolean-like values."""
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()

    for col in df.columns:
        if df[col].dtype == "object":
            lowered = df[col].astype(str).str.strip().str.lower()
            boolean_like = lowered.dropna().isin(["true", "false", "1", "0", "yes", "no"]).all()
            if boolean_like:
                df[col] = lowered.isin(["true", "1", "yes"])

    return df


def select_features(df: pd.DataFrame) -> list[str]:
    """Select score-driver features while avoiding player IDs and date leakage."""
    features = [col for col in PREFERRED_FEATURES if col in df.columns and col not in EXCLUDE_COLUMNS]
    if not features:
        features = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    return features


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature lists."""
    numeric_features = [
        col for col in features if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])
    ]
    categorical_features = [col for col in features if col not in numeric_features]
    return numeric_features, categorical_features


def make_random_forest_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Build a mixed-type random forest pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=30)),
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
                    n_estimators=250,
                    min_samples_leaf=20,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_ridge_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Build a standardized linear model for directional effects."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=30)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", Ridge(alpha=10.0))])


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Return compact regression quality metrics."""
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": float(np.sqrt(mse)),
    }


def build_permutation_importance(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
) -> pd.DataFrame:
    """Measure score impact by shuffling each original input column."""
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="r2",
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "variable": features,
            "importance_mean_r2_drop": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean_r2_drop", ascending=False)


def build_linear_coefficients(model: Pipeline, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    """Aggregate encoded Ridge coefficients back to original variables."""
    preprocessor: ColumnTransformer = model.named_steps["preprocess"]
    coefficients = model.named_steps["model"].coef_

    encoded_names: list[str] = []
    encoded_names.extend(numeric_features)
    if categorical_features:
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_names.extend(onehot.get_feature_names_out(categorical_features).tolist())

    rows = []
    for name, coef in zip(encoded_names, coefficients):
        source = name
        for cat_col in categorical_features:
            if name.startswith(f"{cat_col}_"):
                source = cat_col
                break
        rows.append({"variable": source, "encoded_feature": name, "coefficient": coef})

    encoded = pd.DataFrame(rows)
    grouped = (
        encoded.groupby("variable", dropna=False)
        .agg(
            max_abs_coefficient=("coefficient", lambda x: float(np.abs(x).max())),
            mean_abs_coefficient=("coefficient", lambda x: float(np.abs(x).mean())),
            strongest_encoded_feature=("encoded_feature", lambda x: x.iloc[np.abs(encoded.loc[x.index, "coefficient"]).argmax()]),
            strongest_coefficient=("coefficient", lambda x: float(x.iloc[np.abs(x).argmax()])),
        )
        .reset_index()
        .sort_values("max_abs_coefficient", ascending=False)
    )
    return grouped


def build_numeric_correlations(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    """Calculate Pearson and Spearman correlations with score."""
    rows = []
    for col in numeric_features:
        valid = df[[col, TARGET]].dropna()
        if valid[col].nunique() <= 1:
            continue
        rows.append(
            {
                "variable": col,
                "pearson_corr": valid[col].corr(valid[TARGET], method="pearson"),
                "spearman_corr": valid[col].corr(valid[TARGET], method="spearman"),
                "non_null_rows": len(valid),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["abs_spearman_corr"] = table["spearman_corr"].abs()
    return table.sort_values("abs_spearman_corr", ascending=False)


def build_categorical_effects(df: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
    """Compare category average scores against the global average."""
    global_mean = df[TARGET].mean()
    rows = []
    for col in categorical_features:
        grouped = (
            df.assign(_category=df[col].fillna("Missing").astype(str))
            .groupby("_category", dropna=False)[TARGET]
            .agg(["count", "mean"])
            .reset_index()
        )
        grouped = grouped[grouped["count"] >= 30].copy()
        if grouped.empty:
            continue
        grouped["variable"] = col
        grouped["score_lift_vs_average"] = grouped["mean"] - global_mean
        strongest = grouped.iloc[grouped["score_lift_vs_average"].abs().argmax()]
        rows.append(
            {
                "variable": col,
                "strongest_category": strongest["_category"],
                "category_rows": int(strongest["count"]),
                "category_avg_score": strongest["mean"],
                "score_lift_vs_average": strongest["score_lift_vs_average"],
                "global_avg_score": global_mean,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["abs_score_lift"] = table["score_lift_vs_average"].abs()
    return table.sort_values("abs_score_lift", ascending=False)


def write_summary(
    metrics: dict[str, float],
    feature_count: int,
    row_count: int,
    importance: pd.DataFrame,
    correlations: pd.DataFrame,
    categorical_effects: pd.DataFrame,
) -> None:
    """Write a markdown summary of the most important findings."""
    lines = [
        "# Score Variable Impact Analysis",
        "",
        f"- Dataset rows used: {row_count:,}",
        f"- Variables tested: {feature_count}",
        f"- Random forest test R2: {metrics['r2']:.3f}",
        f"- Random forest test MAE: {metrics['mae']:.2f}",
        f"- Random forest test RMSE: {metrics['rmse']:.2f}",
        "",
        "## Most Important Variables",
        "",
    ]

    for _, row in importance.head(15).iterrows():
        lines.append(
            f"- {row['variable']}: R2 drop {row['importance_mean_r2_drop']:.4f} "
            f"(std {row['importance_std']:.4f})"
        )

    if not correlations.empty:
        lines.extend(["", "## Strongest Numeric Relationships", ""])
        for _, row in correlations.head(10).iterrows():
            lines.append(
                f"- {row['variable']}: Spearman {row['spearman_corr']:.3f}, "
                f"Pearson {row['pearson_corr']:.3f}"
            )

    if not categorical_effects.empty:
        lines.extend(["", "## Strongest Category Score Gaps", ""])
        for _, row in categorical_effects.head(10).iterrows():
            lines.append(
                f"- {row['variable']} = {row['strongest_category']}: "
                f"{row['score_lift_vs_average']:+.2f} points vs average"
            )

    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- `TM Goals`, `TM Assists`, cards, minutes, and lineup status are match outcomes or usage signals, so they explain score strongly but are not all pre-match predictors.",
            "- `permutation_importance` is usually the best table for overall impact because it captures non-linear effects and interactions.",
            "- `linear_grouped_coefficients.csv` is useful for direction, but categorical variables are one-hot encoded and grouped back to their source variable.",
        ]
    )

    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    features = select_features(df)
    numeric_features, categorical_features = split_feature_types(df, features)

    x = df[features].copy()
    y = df[TARGET].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    rf_model = make_random_forest_pipeline(numeric_features, categorical_features)
    rf_model.fit(x_train, y_train)
    metrics = evaluate_model(rf_model, x_test, y_test)
    importance = build_permutation_importance(rf_model, x_test, y_test, features)

    ridge_model = make_ridge_pipeline(numeric_features, categorical_features)
    ridge_model.fit(x_train, y_train)
    coefficients = build_linear_coefficients(ridge_model, numeric_features, categorical_features)

    correlations = build_numeric_correlations(df, numeric_features)
    categorical_effects = build_categorical_effects(df, categorical_features)

    importance.to_csv(OUTPUT_DIR / "model_permutation_importance.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "linear_grouped_coefficients.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "numeric_correlations.csv", index=False)
    categorical_effects.to_csv(OUTPUT_DIR / "categorical_effects.csv", index=False)
    write_summary(metrics, len(features), len(df), importance, correlations, categorical_effects)

    print(f"Rows used: {len(df):,}")
    print(f"Variables tested: {len(features)}")
    print(f"Random forest R2: {metrics['r2']:.3f}")
    print(f"Random forest MAE: {metrics['mae']:.2f}")
    print("\nTop variables by permutation importance:")
    print(importance.head(15).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
