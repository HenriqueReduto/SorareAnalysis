"""Command-line runner for the Sorare score analysis project."""

from __future__ import annotations

import argparse
from pathlib import Path

from analyse_scores import build_rankings, write_summary
from clean_data import clean_scores
from feature_engineering import add_features
from inspect_dataset import inspect_dataset
from load_data import load_data
from model_predictions import train_baseline_model
from visualise import save_charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse Sorare football player scores.")
    parser.add_argument("--input", required=True, help="Path to CSV or Excel file.")
    parser.add_argument("--outputs", default="outputs", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.outputs)
    charts_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    raw = load_data(args.input)

    print("\nSTEP 0 - Dataset inspection")
    inspect_dataset(raw)

    cleaned, reports = clean_scores(raw)
    cleaned.to_csv(output_dir / "cleaned_sorare_scores.csv", index=False)
    for name, report in reports.items():
        if not report.empty:
            report.to_csv(output_dir / f"{name}.csv", index=False)
            print(f"Report saved: {name}.csv ({len(report)} rows)")

    engineered = add_features(cleaned)
    engineered.to_csv(output_dir / "engineered_sorare_scores.csv", index=False)

    tables = build_rankings(engineered)
    tables["player_rankings"].to_csv(output_dir / "player_rankings.csv", index=False)
    tables["best_value_players"].to_csv(output_dir / "value_rankings.csv", index=False)
    write_summary(output_dir / "analysis_summary.md", tables)

    save_charts(engineered, charts_dir)

    predictions, metrics = train_baseline_model(engineered)
    predictions.to_csv(output_dir / "prediction_results.csv", index=False)
    with open(output_dir / "model_metrics.md", "w", encoding="utf-8") as f:
        f.write("# Model Metrics\n\n")
        for key, value in metrics.items():
            f.write(f"- {key}: {value}\n")

    print(f"\nDone. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
