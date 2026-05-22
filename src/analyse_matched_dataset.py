"""Small analysis script for the final Sorare + Transfermarkt matched dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("sorare_outputs/club_player_gw_matched_scores.csv")
OUTPUT_DIR = Path("outputs/matched_analysis")


def load_matched_dataset(path: Path = INPUT_FILE) -> pd.DataFrame:
    """Load and type the matched dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Missing matched dataset: {path}")

    df = pd.read_csv(path)
    df["Game Date"] = pd.to_datetime(df["Game Date"], errors="coerce")
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

    for col in ["TM Goals", "TM Assists", "TM Minutes Played"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["Had Goal", "Had Assist", "Had Goal Contribution", "Matched In Dataset Club"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])

    return df


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build compact analysis tables."""
    lineup_summary = (
        df.groupby("Lineup Status", dropna=False)
        .agg(
            rows=("Score", "count"),
            avg_score=("Score", "mean"),
            median_score=("Score", "median"),
            ceiling=("Score", "max"),
            goals=("TM Goals", "sum"),
            assists=("TM Assists", "sum"),
        )
        .sort_values("avg_score", ascending=False)
        .reset_index()
    )

    player_summary = (
        df[df["Matched In Dataset Club"]]
        .groupby(["Matched Club", "Player", "Position"], dropna=False)
        .agg(
            games=("Score", "count"),
            avg_score=("Score", "mean"),
            median_score=("Score", "median"),
            ceiling=("Score", "max"),
            floor=("Score", "min"),
            volatility=("Score", "std"),
            goals=("TM Goals", "sum"),
            assists=("TM Assists", "sum"),
        )
        .reset_index()
    )
    player_summary = player_summary[player_summary["games"] >= 5].sort_values(
        ["avg_score", "games"],
        ascending=[False, False],
    )

    goal_contribution_summary = (
        df.groupby("Had Goal Contribution", dropna=False)
        .agg(
            rows=("Score", "count"),
            avg_score=("Score", "mean"),
            median_score=("Score", "median"),
            ceiling=("Score", "max"),
        )
        .reset_index()
    )

    club_summary = (
        df[df["Matched In Dataset Club"]]
        .groupby("Matched Club", dropna=False)
        .agg(
            rows=("Score", "count"),
            players=("Player", "nunique"),
            avg_score=("Score", "mean"),
            goals=("TM Goals", "sum"),
            assists=("TM Assists", "sum"),
        )
        .sort_values("avg_score", ascending=False)
        .reset_index()
    )

    return {
        "lineup_summary": lineup_summary,
        "player_summary": player_summary,
        "goal_contribution_summary": goal_contribution_summary,
        "club_summary": club_summary,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_matched_dataset()
    tables = build_tables(df)

    print("Dataset shape:", df.shape)
    print("Players:", df["Player Slug"].nunique())
    print("Matched rows:", int(df["Matched In Dataset Club"].sum()))
    print("Unmatched rows:", int((~df["Matched In Dataset Club"]).sum()))

    for name, table in tables.items():
        out = OUTPUT_DIR / f"{name}.csv"
        table.to_csv(out, index=False)
        print(f"\n{name} saved to {out}")
        print(table.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
