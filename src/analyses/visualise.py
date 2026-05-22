"""Visualisations for Sorare score analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def save_charts(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate and save core charts."""
    charts_dir = Path(output_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    sns.histplot(df["sorare_score"].dropna(), bins=30, kde=True)
    plt.title("Score Distribution")
    plt.tight_layout()
    plt.savefig(charts_dir / "score_distribution_histogram.png", dpi=150)
    plt.close()

    if "position" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x="position", y="sorare_score")
        plt.title("Scores by Position")
        plt.tight_layout()
        plt.savefig(charts_dir / "scores_by_position_boxplot.png", dpi=150)
        plt.close()

    if {"score_ceiling", "consistency_index"}.issubset(df.columns):
        player_level = df.drop_duplicates("player_id")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=player_level, x="consistency_index", y="score_ceiling", hue="position" if "position" in player_level else None)
        plt.title("Consistency vs Ceiling")
        plt.tight_layout()
        plt.savefig(charts_dir / "consistency_vs_ceiling_scatter.png", dpi=150)
        plt.close()

    if {"position", "gameweek"}.issubset(df.columns):
        heatmap_data = df.pivot_table(index="position", columns="gameweek", values="sorare_score", aggfunc="mean")
        plt.figure(figsize=(14, 4))
        sns.heatmap(heatmap_data, cmap="viridis")
        plt.title("Average Score by Position and Gameweek")
        plt.tight_layout()
        plt.savefig(charts_dir / "avg_score_position_gameweek_heatmap.png", dpi=150)
        plt.close()

    value_col = "price_eur" if "price_eur" in df.columns else None
    if value_col:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=value_col, y="sorare_score", hue="position" if "position" in df else None)
        plt.title("Score vs Market Value")
        plt.tight_layout()
        plt.savefig(charts_dir / "score_vs_market_value_scatter.png", dpi=150)
        plt.close()
