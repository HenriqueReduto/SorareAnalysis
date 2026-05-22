"""Data loading and caching for dashboard CSV artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .helpers import OUTPUTS_DIR


EXPECTED_FILES = {
    "final": "final_player_analysis_dataset.csv",
    "rankings": "player_rankings.csv",
    "best": "best_players.csv",
    "feature_importance": "feature_importance.csv",
    "overperformers": "overperformers.csv",
    "underperformers": "underperformers.csv",
    "outliers": "outliers.csv",
    "unmatched": "unmatched_transfermarkt_players.csv",
}

OPTIONAL_FILES = {
    "shap": "shap_importance.csv",
    "correlations": "score_correlations.csv",
}


@dataclass(frozen=True)
class DashboardData:
    """Container for all loaded dashboard datasets."""

    frames: dict[str, pd.DataFrame]
    warnings: list[str]

    def get(self, key: str) -> pd.DataFrame:
        return self.frames.get(key, pd.DataFrame()).copy()


def _read_csv(path) -> pd.DataFrame:
    """Read CSV defensively while letting pandas infer numeric columns."""
    return pd.read_csv(path)


def load_dashboard_data(outputs_dir=OUTPUTS_DIR) -> DashboardData:
    """Load all known analysis outputs and keep running when files are absent."""
    frames: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []

    for key, filename in {**EXPECTED_FILES, **OPTIONAL_FILES}.items():
        path = outputs_dir / filename
        if not path.exists():
            severity = "Missing required file" if key in EXPECTED_FILES else "Optional file not found"
            warnings.append(f"{severity}: `{path}`")
            frames[key] = pd.DataFrame()
            continue
        try:
            frames[key] = _read_csv(path)
        except Exception as exc:  # pragma: no cover - surfaced in UI
            warnings.append(f"Could not load `{path}`: {exc}")
            frames[key] = pd.DataFrame()

    return DashboardData(frames=frames, warnings=warnings)
