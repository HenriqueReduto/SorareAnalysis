r"""End-to-end Sorare football player score analysis workflow.

Run from the project root:

    python src\analyses\player_score_analysis.py

The script inspects available project data, enriches Sorare player rows with
Transfermarkt metadata when available, creates player-level features, models
score drivers, ranks players, detects outliers, and writes CSV/figure/report
outputs to the top-level outputs/ folder.
"""

from __future__ import annotations

import gzip
import io
import math
import re
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(f"matplotlib and seaborn are required for figures: {exc}") from exc

try:
    from scipy import stats
except Exception:  # pragma: no cover - optional fallback
    stats = None

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except Exception as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(f"scikit-learn is required for modelling: {exc}") from exc

try:
    import requests
except Exception:  # pragma: no cover - optional download support
    requests = None

try:
    import shap
except Exception:  # pragma: no cover - SHAP is optional
    shap = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TRANSFERMARKT_DIR = PROJECT_ROOT / "data_preparation" / "datasets" / "transfermarkt"
TRANSFERMARKT_CACHE = PROJECT_ROOT / "data_preparation" / "cache" / "transfermarkt"
REMOTE_PLAYERS_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz"
RANDOM_STATE = 42


@dataclass
class ColumnMap:
    player: str | None = None
    player_id: str | None = None
    tm_player_id: str | None = None
    team: str | None = None
    league: str | None = None
    position: str | None = None
    score: str | None = None
    minutes: str | None = None
    game_date: str | None = None


def normalise_name(value: object) -> str:
    """Normalise names for cross-source matching."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_id(value: object) -> str:
    """Normalise numeric/string IDs so 123 and 123.0 match."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_to_actual = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    return None


def likely_columns(df: pd.DataFrame) -> ColumnMap:
    columns = list(df.columns)
    score_candidates = [
        c
        for c in columns
        if any(token in c.lower() for token in ["score", "rating", "points", "performance"])
        and not any(token in c.lower() for token in ["difficulty", "expected", "adjusted", "z_"])
    ]
    score = None
    if score_candidates:
        preferred = ["Score", "avg_score", "Average_Player_Score", "Average_Score", "SO5 Score"]
        score = first_existing(columns, preferred) or score_candidates[0]

    return ColumnMap(
        player=first_existing(columns, ["Player", "player", "player_name", "tm_player_name", "name"]),
        player_id=first_existing(columns, ["Player Slug", "player_slug", "player_id"]),
        tm_player_id=first_existing(columns, ["tm_player_id", "player_id", "transfermarkt_player_id"]),
        team=first_existing(columns, ["Selected Club", "Club", "Matched Club", "team", "club", "transfermarkt_club_name"]),
        league=first_existing(columns, ["league", "competition_id", "Competition", "competition"]),
        position=first_existing(columns, ["Position", "position", "main_position"]),
        score=score,
        minutes=first_existing(columns, ["TM Minutes Played", "TM Appearance Minutes", "minutes_played", "minutes"]),
        game_date=first_existing(columns, ["Game Date", "Matched Match Date", "match_date", "date"]),
    )


def inspect_workspace() -> list[Path]:
    print("\n=== Workspace files and folders ===")
    files = sorted(PROJECT_ROOT.rglob("*"))
    for path in files:
        rel = path.relative_to(PROJECT_ROOT)
        if ".git" in rel.parts:
            continue
        marker = "/" if path.is_dir() else ""
        print(f"{rel}{marker}")
    data_files = [
        p
        for p in files
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".parquet", ".xlsx", ".xls", ".ipynb", ".py"}
    ]
    print("\n=== Detected CSV/JSON/parquet/API-related files ===")
    for path in data_files:
        print(f"{path.relative_to(PROJECT_ROOT)} ({path.stat().st_size:,} bytes)")
    return data_files


def inspect_dataset(df: pd.DataFrame, name: str) -> None:
    cmap = likely_columns(df)
    print(f"\n=== Dataset inspection: {name} ===")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print("\ndata types:")
    print(df.dtypes.to_string())
    print("\nmissing values:")
    print(df.isna().sum().sort_values(ascending=False).to_string())
    print(f"\nduplicate rows: {int(df.duplicated().sum())}")
    print("\nsample rows:")
    print(df.head(5).to_string())
    print("\ndescriptive statistics:")
    print(df.describe(include="all").transpose().head(80).to_string())
    print(f"\nmemory usage MB: {df.memory_usage(deep=True).sum() / 1_000_000:.2f}")
    print(f"numerical columns: {df.select_dtypes(include=np.number).columns.tolist()}")
    print(f"categorical columns: {df.select_dtypes(exclude=np.number).columns.tolist()}")
    print(f"likely player ID/name columns: {[c for c in [cmap.player_id, cmap.tm_player_id, cmap.player] if c]}")
    score_candidates = [
        c
        for c in df.columns
        if any(token in c.lower() for token in ["score", "rating", "points", "performance"])
    ]
    print(f"likely target score columns: {score_candidates}")
    print(f"selected target column: {cmap.score}")


def choose_main_dataset(data_files: list[Path]) -> Path:
    preferred = [
        PROJECT_ROOT / "data_preparation" / "datasets" / "combined" / "sorare_transfermarkt_context.csv",
        PROJECT_ROOT / "data_preparation" / "datasets" / "combined" / "club_player_gw_matched_scores.csv",
        PROJECT_ROOT / "data_preparation" / "datasets" / "sorare" / "all_players_scores_long.csv",
        PROJECT_ROOT / "analyses" / "matched_dataset" / "player_summary.csv",
    ]
    for path in preferred:
        if path.exists():
            print(f"\nSelected main player dataset: {path.relative_to(PROJECT_ROOT)}")
            return path
    csvs = [p for p in data_files if p.suffix.lower() == ".csv"]
    if not csvs:
        raise FileNotFoundError("No CSV data files were found in the project.")
    selected = max(csvs, key=lambda p: p.stat().st_size)
    print(f"\nSelected largest CSV as fallback main dataset: {selected.relative_to(PROJECT_ROOT)}")
    return selected


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False, nrows=nrows)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table type: {path}")


def parse_height_cm(series: pd.Series) -> pd.Series:
    def convert(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower().replace(",", ".")
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return np.nan
        number = float(match.group(1))
        if "m" in text and number < 3:
            number *= 100
        return number

    return series.map(convert)


def compute_age(date_of_birth: pd.Series, reference_date: pd.Timestamp) -> pd.Series:
    dob = pd.to_datetime(date_of_birth, errors="coerce", utc=True)
    return (reference_date.tz_localize("UTC") - dob).dt.days / 365.25


def load_transfermarkt_metadata(reference_date: pd.Timestamp) -> pd.DataFrame:
    """Load local Transfermarkt metadata and add public players metadata if available."""
    frames: list[pd.DataFrame] = []
    for path in sorted(TRANSFERMARKT_DIR.glob("*")):
        if path.suffix.lower() not in {".csv", ".json", ".parquet", ".xlsx", ".xls"}:
            continue
        try:
            df = read_table(path)
            df["source_file"] = str(path.relative_to(PROJECT_ROOT))
            frames.append(df)
            print(f"Loaded Transfermarkt file: {path.relative_to(PROJECT_ROOT)} {df.shape}")
        except Exception as exc:
            print(f"Warning: could not read Transfermarkt file {path}: {exc}")

    players_cache = TRANSFERMARKT_CACHE / "players.csv"
    if players_cache.exists():
        try:
            players = pd.read_csv(players_cache, low_memory=False)
            players["source_file"] = str(players_cache.relative_to(PROJECT_ROOT))
            frames.append(players)
            print(f"Loaded cached Transfermarkt players metadata: {players.shape}")
        except Exception as exc:
            print(f"Warning: could not read cached Transfermarkt players table: {exc}")
    elif requests is not None:
        try:
            print("Attempting optional Transfermarkt players metadata download for age/height...")
            response = requests.get(REMOTE_PLAYERS_URL, headers={"User-Agent": "Mozilla/5.0 sorare-analysis"}, timeout=45)
            response.raise_for_status()
            TRANSFERMARKT_CACHE.mkdir(parents=True, exist_ok=True)
            players = pd.read_csv(gzip.GzipFile(fileobj=io.BytesIO(response.content)), low_memory=False)
            players.to_csv(players_cache, index=False)
            players["source_file"] = str(players_cache.relative_to(PROJECT_ROOT))
            frames.append(players)
            print(f"Downloaded Transfermarkt players metadata: {players.shape}")
        except Exception as exc:
            print(f"Warning: optional Transfermarkt players metadata download failed: {exc}")

    if not frames:
        return pd.DataFrame()

    standardised = []
    for raw in frames:
        df = raw.copy()
        id_col = first_existing(df.columns, ["tm_player_id", "player_id"])
        name_col = first_existing(df.columns, ["tm_player_name", "player_name", "name"])
        club_col = first_existing(df.columns, ["transfermarkt_club_name", "current_club_name", "club_name"])
        height_col = first_existing(df.columns, ["height_in_cm", "height", "Height"])
        age_col = first_existing(df.columns, ["age", "Age"])
        dob_col = first_existing(df.columns, ["date_of_birth", "birth_date", "Date of birth"])
        if not id_col and not name_col:
            continue
        out = pd.DataFrame(index=df.index)
        out["tm_player_id"] = df[id_col].map(normalise_id) if id_col else pd.NA
        out["tm_player_name"] = df[name_col] if name_col else pd.NA
        out["tm_player_norm"] = out["tm_player_name"].map(normalise_name)
        out["tm_team"] = df[club_col] if club_col else pd.NA
        out["tm_team_norm"] = out["tm_team"].map(normalise_name)
        out["height_cm"] = parse_height_cm(df[height_col]) if height_col else np.nan
        if age_col:
            out["age"] = pd.to_numeric(df[age_col], errors="coerce")
        elif dob_col:
            out["age"] = compute_age(df[dob_col], reference_date)
        else:
            out["age"] = np.nan
        out["metadata_source"] = df.get("source_file", "unknown")
        standardised.append(out)

    if not standardised:
        return pd.DataFrame()
    meta = pd.concat(standardised, ignore_index=True)
    meta = meta.dropna(how="all", subset=["tm_player_id", "tm_player_norm"])
    meta = (
        meta.sort_values(["height_cm", "age"], na_position="last")
        .drop_duplicates(["tm_player_id", "tm_player_norm", "tm_team_norm"], keep="first")
        .reset_index(drop=True)
    )
    meta["age_flag_impossible"] = meta["age"].notna() & ~meta["age"].between(14, 50)
    meta["height_flag_impossible"] = meta["height_cm"].notna() & ~meta["height_cm"].between(140, 220)
    meta.loc[meta["age_flag_impossible"], "age"] = np.nan
    meta.loc[meta["height_flag_impossible"], "height_cm"] = np.nan
    return meta


def clean_dataset(df: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
    cleaned = df.drop_duplicates().copy()
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})
    if cmap.player:
        cleaned["player_name_standardized"] = cleaned[cmap.player].map(normalise_name)
    if cmap.team:
        cleaned["team_standardized"] = cleaned[cmap.team].map(normalise_name)
    if cmap.score:
        cleaned[cmap.score] = pd.to_numeric(cleaned[cmap.score], errors="coerce")
    if cmap.minutes:
        cleaned[cmap.minutes] = pd.to_numeric(cleaned[cmap.minutes], errors="coerce")
        cleaned.loc[cleaned[cmap.minutes] < 0, cmap.minutes] = np.nan
        cleaned.loc[cleaned[cmap.minutes] > 130, cmap.minutes] = np.nan
    for col in cleaned.columns:
        if any(token in col.lower() for token in ["goals", "assists", "cards", "position", "ppg", "rate", "difficulty"]):
            if cleaned[col].dtype == "object":
                converted = pd.to_numeric(cleaned[col], errors="coerce")
                if converted.notna().mean() > 0.7:
                    cleaned[col] = converted
    return cleaned


def enrich_with_transfermarkt(df: pd.DataFrame, cmap: ColumnMap) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    reference = pd.Timestamp.today().normalize()
    if cmap.game_date and cmap.game_date in df:
        dates = pd.to_datetime(df[cmap.game_date], errors="coerce", utc=True)
        if dates.notna().any():
            reference = dates.max().tz_convert(None).normalize()

    meta = load_transfermarkt_metadata(reference)
    enriched = df.copy()
    enriched["age"] = np.nan
    enriched["height_cm"] = np.nan
    enriched["transfermarkt_match_confidence"] = "unmatched"
    enriched["transfermarkt_match_method"] = "none"
    enriched["transfermarkt_metadata_source"] = pd.NA
    enriched["age_flag_impossible"] = False
    enriched["height_flag_impossible"] = False

    if meta.empty:
        print("Warning: no Transfermarkt metadata with age/height was available. Age and height will remain missing.")
        unmatched = enriched[[c for c in [cmap.player, cmap.team, cmap.position] if c]].drop_duplicates()
        return enriched, unmatched, {"matched": 0, "unmatched": len(unmatched)}

    enriched["_row_id"] = np.arange(len(enriched))
    enriched["_tm_id_key"] = enriched[cmap.tm_player_id].map(normalise_id) if cmap.tm_player_id else pd.NA
    enriched["_player_norm"] = enriched[cmap.player].map(normalise_name) if cmap.player else ""
    enriched["_team_norm"] = enriched[cmap.team].map(normalise_name) if cmap.team else ""

    exact = enriched.merge(
        meta.dropna(subset=["tm_player_id"]).drop_duplicates("tm_player_id"),
        left_on="_tm_id_key",
        right_on="tm_player_id",
        how="left",
        suffixes=("", "_meta"),
    )
    matched_mask = exact["age_meta"].notna() | exact["height_cm_meta"].notna()
    for col in ["age", "height_cm", "metadata_source", "age_flag_impossible", "height_flag_impossible"]:
        meta_col = f"{col}_meta"
        if meta_col in exact:
            exact.loc[matched_mask, col if col != "metadata_source" else "transfermarkt_metadata_source"] = exact.loc[
                matched_mask, meta_col
            ].values
        elif col == "metadata_source" and "metadata_source" in exact:
            exact.loc[matched_mask, "transfermarkt_metadata_source"] = exact.loc[matched_mask, "metadata_source"].values
    exact.loc[matched_mask, "transfermarkt_match_confidence"] = "exact_id"
    exact.loc[matched_mask, "transfermarkt_match_method"] = "tm_player_id"
    enriched = exact[[c for c in exact.columns if not c.endswith("_meta")]].copy()

    unmatched_mask = enriched["transfermarkt_match_confidence"].eq("unmatched")
    meta_name_team = meta.dropna(subset=["tm_player_norm"]).drop_duplicates(["tm_player_norm", "tm_team_norm"])
    name_team = enriched[unmatched_mask].merge(
        meta_name_team,
        left_on=["_player_norm", "_team_norm"],
        right_on=["tm_player_norm", "tm_team_norm"],
        how="left",
        suffixes=("", "_meta"),
    )
    nt_ids = set(name_team.loc[name_team["age_meta"].notna() | name_team["height_cm_meta"].notna(), "_row_id"])
    if nt_ids:
        keyed = name_team.set_index("_row_id")
        for row_id in nt_ids:
            row = keyed.loc[row_id]
            idx = enriched["_row_id"].eq(row_id)
            enriched.loc[idx, "age"] = row.get("age_meta")
            enriched.loc[idx, "height_cm"] = row.get("height_cm_meta")
            enriched.loc[idx, "transfermarkt_metadata_source"] = row.get(
                "metadata_source_meta", row.get("metadata_source")
            )
            enriched.loc[idx, "age_flag_impossible"] = bool(row.get("age_flag_impossible_meta", False))
            enriched.loc[idx, "height_flag_impossible"] = bool(row.get("height_flag_impossible_meta", False))
            enriched.loc[idx, "transfermarkt_match_confidence"] = "high"
            enriched.loc[idx, "transfermarkt_match_method"] = "name_team"

    unmatched_mask = enriched["transfermarkt_match_confidence"].eq("unmatched")
    meta_name = meta.dropna(subset=["tm_player_norm"]).drop_duplicates("tm_player_norm")
    name_only = enriched[unmatched_mask].merge(
        meta_name,
        left_on="_player_norm",
        right_on="tm_player_norm",
        how="left",
        suffixes=("", "_meta"),
    )
    no_ids = set(name_only.loc[name_only["age_meta"].notna() | name_only["height_cm_meta"].notna(), "_row_id"])
    if no_ids:
        keyed = name_only.set_index("_row_id")
        for row_id in no_ids:
            row = keyed.loc[row_id]
            idx = enriched["_row_id"].eq(row_id)
            enriched.loc[idx, "age"] = row.get("age_meta")
            enriched.loc[idx, "height_cm"] = row.get("height_cm_meta")
            enriched.loc[idx, "transfermarkt_metadata_source"] = row.get(
                "metadata_source_meta", row.get("metadata_source")
            )
            enriched.loc[idx, "age_flag_impossible"] = bool(row.get("age_flag_impossible_meta", False))
            enriched.loc[idx, "height_flag_impossible"] = bool(row.get("height_flag_impossible_meta", False))
            enriched.loc[idx, "transfermarkt_match_confidence"] = "medium"
            enriched.loc[idx, "transfermarkt_match_method"] = "name"

    # Fuzzy matching is applied only at unique player level to keep runtime bounded.
    unmatched_players = (
        enriched[enriched["transfermarkt_match_confidence"].eq("unmatched")]
        .drop_duplicates(["_player_norm", "_team_norm"])
        [["_player_norm", "_team_norm"]]
    )
    meta_candidates = meta[meta["age"].notna() | meta["height_cm"].notna()].drop_duplicates("tm_player_norm").copy()
    meta_candidates["_first_token"] = meta_candidates["tm_player_norm"].str.split().str[0].fillna("")
    meta_by_first_token = {key: value for key, value in meta_candidates.groupby("_first_token")}
    fuzzy_updates = {}
    for _, player_row in unmatched_players.head(1000).iterrows():
        name = player_row["_player_norm"]
        if not name:
            continue
        first_token = name.split()[0]
        pool = meta_by_first_token.get(first_token, meta_candidates.head(0))
        if player_row["_team_norm"]:
            same_team = pool[pool["tm_team_norm"].eq(player_row["_team_norm"])]
            if not same_team.empty:
                pool = same_team
        if len(pool) > 350:
            pool = pool.head(350)
        best_score, best = 0.0, None
        for _, cand in pool.iterrows():
            score = SequenceMatcher(None, name, cand["tm_player_norm"]).ratio()
            if score > best_score:
                best_score, best = score, cand
        if best is not None and best_score >= 0.90:
            fuzzy_updates[(name, player_row["_team_norm"])] = best
    for (name, team), row in fuzzy_updates.items():
        idx = (
            enriched["transfermarkt_match_confidence"].eq("unmatched")
            & enriched["_player_norm"].eq(name)
            & enriched["_team_norm"].eq(team)
        )
        enriched.loc[idx, "age"] = row.get("age")
        enriched.loc[idx, "height_cm"] = row.get("height_cm")
        enriched.loc[idx, "transfermarkt_metadata_source"] = row.get("metadata_source")
        enriched.loc[idx, "transfermarkt_match_confidence"] = "fuzzy_high"
        enriched.loc[idx, "transfermarkt_match_method"] = "fuzzy_name_team"

    unmatched_cols = [c for c in [cmap.player, cmap.team, cmap.position] if c]
    unmatched = enriched[enriched["transfermarkt_match_confidence"].eq("unmatched")][unmatched_cols].drop_duplicates()
    summary = {
        "matched": int((~enriched["transfermarkt_match_confidence"].eq("unmatched")).sum()),
        "unmatched": int(enriched["transfermarkt_match_confidence"].eq("unmatched").sum()),
        "exact_id": int(enriched["transfermarkt_match_confidence"].eq("exact_id").sum()),
        "high": int(enriched["transfermarkt_match_confidence"].eq("high").sum()),
        "medium": int(enriched["transfermarkt_match_confidence"].eq("medium").sum()),
        "fuzzy_high": int(enriched["transfermarkt_match_confidence"].eq("fuzzy_high").sum()),
    }
    return enriched.drop(columns=["_row_id", "_tm_id_key", "_player_norm", "_team_norm"], errors="ignore"), unmatched, summary


def aggregate_player_dataset(df: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
    if not cmap.player or not cmap.score:
        raise ValueError("A player and score column are required for player-level analysis.")
    group_cols = [
        c
        for c in ["player_name_standardized", "team_standardized", cmap.position]
        if c and c in df.columns
    ]
    if not group_cols:
        group_cols = [c for c in [cmap.player_id, cmap.player, cmap.position, cmap.team] if c]
    minutes = cmap.minutes
    score = cmap.score
    work = df.copy()
    if minutes:
        work["_minutes"] = work[minutes].fillna(0)
    else:
        work["_minutes"] = np.nan
    if cmap.league:
        work["_league"] = work[cmap.league]
    else:
        work["_league"] = pd.NA

    # Some Sorare extracts can contain the same real player under more than one
    # slug. For player-level analysis, keep one row per player/team/position/game
    # identity, preferring the row with the richer or higher-scoring observation.
    dedupe_cols = [c for c in group_cols + ([cmap.game_date] if cmap.game_date else []) if c in work.columns]
    if dedupe_cols:
        sort_cols = [score]
        ascending = [False]
        if "Matched In Dataset Club" in work.columns:
            sort_cols.append("Matched In Dataset Club")
            ascending.append(False)
        if minutes:
            sort_cols.append(minutes)
            ascending.append(False)
        work = work.sort_values(sort_cols, ascending=ascending).drop_duplicates(dedupe_cols, keep="first")

    def mode_value(series: pd.Series) -> object:
        mode = series.dropna().mode()
        return mode.iat[0] if not mode.empty else pd.NA

    named_aggs = {
        cmap.player: (cmap.player, mode_value),
        "appearances": (score, "count"),
        "avg_score": (score, "mean"),
        "median_score": (score, "median"),
        "max_score": (score, "max"),
        "min_score": (score, "min"),
        "score_std": (score, "std"),
        "total_score": (score, "sum"),
        "total_minutes": ("_minutes", "sum"),
        "avg_minutes": ("_minutes", "mean"),
        "age": ("age", "median"),
        "height_cm": ("height_cm", "median"),
        "leagues_played": ("_league", lambda s: ", ".join(sorted(str(v) for v in s.dropna().unique()))),
        "transfermarkt_match_confidence": (
            "transfermarkt_match_confidence",
            lambda s: s.mode().iat[0] if not s.mode().empty else "unmatched",
        ),
    }
    if cmap.player_id:
        named_aggs[cmap.player_id] = (cmap.player_id, mode_value)
        named_aggs["player_ids_seen"] = (cmap.player_id, lambda s: ", ".join(sorted(str(v) for v in s.dropna().unique())))
    if cmap.team:
        named_aggs[cmap.team] = (cmap.team, mode_value)
    if cmap.league:
        named_aggs[cmap.league] = ("_league", mode_value)

    agg = (
        work.groupby(group_cols, dropna=False)
        .agg(**named_aggs)
        .reset_index()
    )
    agg["score_std"] = agg["score_std"].fillna(0)
    agg["score_per_90"] = np.where(agg["total_minutes"] > 0, agg["total_score"] / agg["total_minutes"] * 90, np.nan)
    minutes_factor = np.clip(agg["total_minutes"].fillna(0) / 900, 0.15, 1.0)
    agg["minutes_adjusted_score"] = agg["avg_score"] * minutes_factor
    if cmap.position:
        agg["position_adjusted_score"] = agg["avg_score"] - agg.groupby(cmap.position)["avg_score"].transform("median")
        agg["position_adjusted_height"] = agg["height_cm"] - agg.groupby(cmap.position)["height_cm"].transform("median")
    else:
        agg["position_adjusted_score"] = agg["avg_score"] - agg["avg_score"].median()
        agg["position_adjusted_height"] = agg["height_cm"] - agg["height_cm"].median()
    agg["consistency_score"] = agg["avg_score"] / (agg["score_std"] + 1)
    agg["availability_rate"] = np.where(agg["appearances"] > 0, agg["total_minutes"] / (agg["appearances"] * 90), np.nan)
    agg["age_bucket"] = pd.cut(
        agg["age"],
        bins=[0, 21, 24, 28, 32, 60],
        labels=["U21", "22-24", "25-28", "29-32", "33+"],
    )
    agg["young_player_flag"] = agg["age"].notna() & (agg["age"] <= 23)
    agg["peak_age_flag"] = agg["age"].notna() & agg["age"].between(24, 29)
    agg["height_category"] = pd.cut(
        agg["height_cm"],
        bins=[0, 175, 185, 250],
        labels=["short", "average", "tall"],
    )
    return agg.sort_values("avg_score", ascending=False)


def prepare_model_data(player_df: pd.DataFrame, target: str = "avg_score") -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    exclude = {
        target,
        "median_score",
        "max_score",
        "min_score",
        "score_std",
        "total_score",
        "minutes_adjusted_score",
        "position_adjusted_score",
        "score_per_90",
        "consistency_score",
        "expected_score",
        "score_residual",
    }
    candidates = [c for c in player_df.columns if c not in exclude]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(player_df[c])]
    categorical = [
        c
        for c in candidates
        if not pd.api.types.is_numeric_dtype(player_df[c]) and player_df[c].nunique(dropna=True) <= 50
    ]
    model_df = player_df[[target] + numeric + categorical].dropna(subset=[target]).copy()
    model_df = model_df.replace({pd.NA: np.nan})
    for col in categorical:
        model_df[col] = model_df[col].astype("object")
    model_df = model_df[model_df[target].notna()]
    return model_df[numeric + categorical], model_df[target], numeric, categorical


def run_feature_importance(player_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, Pipeline | None]:
    x, y, numeric, categorical = prepare_model_data(player_df)
    if len(x) < 20 or len(numeric + categorical) == 0:
        print("Warning: not enough data/features for feature importance modelling.")
        return pd.DataFrame(), {}, pd.DataFrame(), None

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=RANDOM_STATE)
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    transformed_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    def source_feature_name(feature: str) -> str:
        if feature in numeric:
            return feature
        for cat_col in categorical:
            if feature == cat_col or feature.startswith(f"{cat_col}_"):
                return cat_col
        return feature

    rf_importance = pd.DataFrame(
        {"feature": transformed_names, "rf_importance": pipe.named_steps["model"].feature_importances_}
    )
    rf_importance["source_feature"] = rf_importance["feature"].map(source_feature_name)
    grouped_rf = rf_importance.groupby("source_feature", as_index=False)["rf_importance"].sum()

    perm = permutation_importance(pipe, x_test, y_test, n_repeats=6, random_state=RANDOM_STATE, n_jobs=-1)
    perm_df = pd.DataFrame(
        {
            "source_feature": x.columns,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    )
    importance = grouped_rf.merge(perm_df, on="source_feature", how="outer").fillna(0)
    importance["combined_rank_score"] = (
        importance["rf_importance"].rank(ascending=False, method="min")
        + importance["permutation_importance_mean"].rank(ascending=False, method="min")
    )
    importance = importance.sort_values(["combined_rank_score", "rf_importance"], ascending=[True, False])

    shap_df = pd.DataFrame()
    if shap is not None:
        try:
            transformed = pipe.named_steps["preprocessor"].transform(x_train.head(250))
            explainer = shap.TreeExplainer(pipe.named_steps["model"])
            values = explainer.shap_values(transformed)
            shap_df = pd.DataFrame({"feature": transformed_names, "mean_abs_shap": np.abs(values).mean(axis=0)})
        except Exception as exc:
            print(f"Warning: SHAP calculation skipped: {exc}")
    return importance, metrics, shap_df, pipe


def correlation_tables(player_df: pd.DataFrame, target: str = "avg_score") -> pd.DataFrame:
    numeric = player_df.select_dtypes(include=np.number).columns.tolist()
    rows = []
    for col in numeric:
        if col == target:
            continue
        pair = player_df[[target, col]].dropna()
        if len(pair) < 5 or pair[col].nunique() <= 1:
            continue
        rows.append(
            {
                "feature": col,
                "pearson": pair[target].corr(pair[col], method="pearson"),
                "spearman": pair[target].corr(pair[col], method="spearman"),
                "n": len(pair),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson", key=lambda s: s.abs(), ascending=False)


def build_rankings(player_df: pd.DataFrame, cmap: ColumnMap) -> dict[str, pd.DataFrame]:
    rank_cols = [c for c in [cmap.player_id, cmap.player, cmap.position, cmap.team, cmap.league, "leagues_played"] if c and c in player_df.columns]
    rank_cols += [
        "avg_score",
        "minutes_adjusted_score",
        "position_adjusted_score",
        "score_per_90",
        "total_minutes",
        "appearances",
        "age",
        "height_cm",
        "consistency_score",
        "availability_rate",
    ]
    eligible = player_df[player_df["appearances"] >= max(3, player_df["appearances"].quantile(0.25))].copy()
    rankings = {
        "player_rankings": player_df[rank_cols].sort_values("minutes_adjusted_score", ascending=False),
        "best_players": eligible[rank_cols].sort_values("avg_score", ascending=False).head(50),
        "best_young_players": eligible[eligible["young_player_flag"]][rank_cols].sort_values("minutes_adjusted_score", ascending=False).head(50),
        "most_consistent_players": eligible[rank_cols].sort_values("consistency_score", ascending=False).head(50),
        "highest_adjusted_score_players": eligible[rank_cols].sort_values("position_adjusted_score", ascending=False).head(50),
    }
    if cmap.position:
        by_pos = (
            eligible.sort_values(["Position" if cmap.position == "Position" else cmap.position, "minutes_adjusted_score"], ascending=[True, False])
            .groupby(cmap.position)
            .head(20)
        )
        rankings["best_players_by_position"] = by_pos[rank_cols]
    return rankings


def outlier_detection(player_df: pd.DataFrame) -> pd.DataFrame:
    df = player_df.copy()
    numeric_cols = ["avg_score", "minutes_adjusted_score", "position_adjusted_score", "total_minutes", "age", "height_cm"]
    for col in numeric_cols:
        if col in df and df[col].notna().sum() > 3:
            std = df[col].std()
            df[f"{col}_zscore"] = (df[col] - df[col].mean()) / std if std else 0
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df[f"{col}_iqr_outlier"] = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    iso_features = [c for c in numeric_cols if c in df and df[c].notna().sum() > 10]
    if iso_features:
        x = df[iso_features].replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.median(numeric_only=True))
        iso = IsolationForest(contamination=min(0.12, max(0.02, 20 / max(len(x), 1))), random_state=RANDOM_STATE)
        df["isolation_forest_outlier"] = iso.fit_predict(x) == -1
    else:
        df["isolation_forest_outlier"] = False
    df["outlier_category"] = ""
    df.loc[df.get("avg_score_zscore", 0) > 2, "outlier_category"] += "unusually high performer; "
    df.loc[df.get("avg_score_zscore", 0) < -2, "outlier_category"] += "unusually low performer; "
    df.loc[df.get("height_cm_iqr_outlier", False), "outlier_category"] += "unusual physical profile; "
    df.loc[(df["avg_score"] > df["avg_score"].quantile(0.85)) & (df["total_minutes"] < df["total_minutes"].quantile(0.25)), "outlier_category"] += "high score with low minutes; "
    df.loc[(df["avg_score"] < df["avg_score"].quantile(0.25)) & (df["total_minutes"] > df["total_minutes"].quantile(0.75)), "outlier_category"] += "low score with high minutes; "
    df.loc[df["isolation_forest_outlier"], "outlier_category"] += "isolation forest anomaly; "
    return df[df["outlier_category"].str.len() > 0].sort_values("outlier_category")


def expected_score_residuals(player_df: pd.DataFrame, pipe: Pipeline | None) -> pd.DataFrame:
    df = player_df.copy()
    x, _, _, _ = prepare_model_data(df)
    if pipe is None or x.empty:
        df["expected_score"] = df["avg_score"].median()
    else:
        df.loc[x.index, "expected_score"] = pipe.predict(x)
        df["expected_score"] = df["expected_score"].fillna(df["avg_score"].median())
    df["score_residual"] = df["avg_score"] - df["expected_score"]
    return df.sort_values("score_residual", ascending=False)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def make_figures(player_df: pd.DataFrame, importance: pd.DataFrame, outliers: pd.DataFrame, cmap: ColumnMap) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    numeric = player_df.select_dtypes(include=np.number)
    if not numeric.empty:
        plt.figure(figsize=(12, 9))
        corr = numeric.corr(numeric_only=True)
        sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
        plt.title("Correlation Heatmap")
        save_plot(FIGURE_DIR / "correlation_heatmap.png")

    if not importance.empty:
        top = importance.head(20)
        plt.figure(figsize=(10, 7))
        sns.barplot(data=top, y="source_feature", x="rf_importance", color="#3274a1")
        plt.title("Random Forest Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        save_plot(FIGURE_DIR / "feature_importance_bar_chart.png")

    plot_df = player_df.dropna(subset=["expected_score", "avg_score"])
    if not plot_df.empty:
        plt.figure(figsize=(8, 7))
        sns.scatterplot(data=plot_df, x="expected_score", y="avg_score", hue=cmap.position if cmap.position else None)
        lims = [min(plot_df["expected_score"].min(), plot_df["avg_score"].min()), max(plot_df["expected_score"].max(), plot_df["avg_score"].max())]
        plt.plot(lims, lims, "--", color="black", linewidth=1)
        plt.title("Expected vs Actual Score")
        plt.xlabel("Expected score")
        plt.ylabel("Actual average score")
        save_plot(FIGURE_DIR / "expected_vs_actual_score_scatterplot.png")

        plt.figure(figsize=(8, 5))
        sns.histplot(plot_df["score_residual"], kde=True, color="#4c78a8")
        plt.title("Residual Distribution")
        plt.xlabel("Actual minus expected score")
        plt.ylabel("Players")
        save_plot(FIGURE_DIR / "residual_distribution.png")

    name_col = cmap.player or "player"
    for file_name, title, data in [
        ("top_players_bar_chart.png", "Top Players by Adjusted Score", player_df.sort_values("minutes_adjusted_score", ascending=False).head(20)),
        ("overperformers_bar_chart.png", "Top Overperformers", player_df.sort_values("score_residual", ascending=False).head(20)),
        ("underperformers_bar_chart.png", "Top Underperformers", player_df.sort_values("score_residual").head(20)),
    ]:
        if name_col in data:
            plt.figure(figsize=(10, 7))
            sns.barplot(data=data, y=name_col, x="minutes_adjusted_score" if "Top Players" in title else "score_residual", color="#59a14f")
            plt.title(title)
            plt.xlabel("Adjusted score" if "Top Players" in title else "Residual")
            plt.ylabel("Player")
            save_plot(FIGURE_DIR / file_name)

    if not outliers.empty:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=player_df, x="total_minutes", y="avg_score", hue="isolation_forest_outlier" if "isolation_forest_outlier" in player_df else None)
        plt.title("Outlier Scatterplot: Minutes vs Score")
        plt.xlabel("Total minutes")
        plt.ylabel("Average score")
        save_plot(FIGURE_DIR / "outlier_minutes_score_scatterplot.png")

    for x_col, file_name, title in [
        ("age", "age_vs_score_plot.png", "Age vs Score"),
        ("height_cm", "height_vs_score_plot.png", "Height vs Score"),
    ]:
        if x_col in player_df and player_df[x_col].notna().any():
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=player_df, x=x_col, y="avg_score", hue=cmap.position if cmap.position else None)
            plt.title(title)
            plt.xlabel(x_col.replace("_", " ").title())
            plt.ylabel("Average score")
            save_plot(FIGURE_DIR / file_name)

    if cmap.position:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=player_df, x=cmap.position, y="position_adjusted_score")
        plt.title("Position-Adjusted Score by Position")
        plt.xlabel("Position")
        plt.ylabel("Position-adjusted score")
        save_plot(FIGURE_DIR / "position_adjusted_score_plot.png")


def write_summary(
    source_path: Path,
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    enriched: pd.DataFrame,
    player_df: pd.DataFrame,
    importance: pd.DataFrame,
    correlations: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
    outliers: pd.DataFrame,
    tm_summary: dict[str, int],
    metrics: dict[str, float],
) -> Path:
    top_features = importance.head(10).to_markdown(index=False) if not importance.empty else "No model feature importance available."
    top_corr = correlations.sort_values("pearson", ascending=False).head(8).to_markdown(index=False) if not correlations.empty else "No correlations available."
    low_corr = correlations.sort_values("pearson").head(8).to_markdown(index=False) if not correlations.empty else "No correlations available."
    best = rankings["best_players"].head(15).to_markdown(index=False) if "best_players" in rankings else "No rankings available."
    over = player_df.sort_values("score_residual", ascending=False).head(15).to_markdown(index=False)
    under = player_df.sort_values("score_residual").head(15).to_markdown(index=False)
    out = outliers.head(20).to_markdown(index=False) if not outliers.empty else "No major outliers detected."
    summary = f"""# Player Score Analysis Summary

## Dataset Overview
- Main dataset: `{source_path.relative_to(PROJECT_ROOT)}`
- Raw shape: {raw.shape}
- Cleaned shape: {cleaned.shape}
- Enriched row shape: {enriched.shape}
- Final player-level shape: {player_df.shape}
- Model metrics: {metrics}

## Cleaning Decisions
- Removed exact duplicate rows.
- Standardized player and team names for matching.
- Converted detected score and minutes columns to numeric.
- Invalid minutes below 0 or above 130 were set missing.
- Age outside 14-50 and height outside 140-220 cm are flagged and excluded from numeric age/height values.
- Rankings use minutes-adjusted and position-adjusted scores to reduce unfair comparisons across availability and roles.

## Transfermarkt Matching Summary
{tm_summary}

## Strongest Score Drivers
{top_features}

## Top Positive Correlations
{top_corr}

## Top Negative Correlations
{low_corr}

## Best Players
{best}

## Overperformers
{over}

## Underperformers
{under}

## Outliers
{out}

## Limitations
- Transfermarkt age and height depend on local metadata or the optional public players table being available.
- Some columns may be post-match variables, so they explain score but should not all be used for pre-match prediction.
- Team and league effects can absorb context that belongs to player role, opposition strength, or selection bias.
- Players with low minutes remain volatile even after minutes adjustment.

## Recommended Next Steps
- Add a pinned Transfermarkt `players.csv` metadata file to `data_preparation/datasets/transfermarkt/` for fully offline age/height enrichment.
- Separate pre-match-only features from post-match explanatory features.
- Add rolling player form and opponent-specific features before building a production forecast.
- Review unmatched players and add manual aliases for high-value misses.
"""
    path = OUTPUT_DIR / "analysis_summary.md"
    path.write_text(summary, encoding="utf-8")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data_files = inspect_workspace()
    print("\n=== Transfermarkt folder inspection ===")
    if TRANSFERMARKT_DIR.exists():
        for path in sorted(TRANSFERMARKT_DIR.iterdir()):
            print(f"{path.relative_to(PROJECT_ROOT)} ({'dir' if path.is_dir() else str(path.stat().st_size) + ' bytes'})")
    else:
        print("Warning: transfermarkt folder does not exist.")

    source_path = choose_main_dataset(data_files)
    raw = read_table(source_path)
    inspect_dataset(raw, source_path.name)
    cmap = likely_columns(raw)
    if not cmap.score:
        raise ValueError("Could not detect a player score target column.")
    print(f"\nTarget selection logic chose `{cmap.score}` from score-like columns because it is the preferred/raw score field.")

    cleaned = clean_dataset(raw, cmap)
    cleaned_path = OUTPUT_DIR / "cleaned_player_dataset.csv"
    cleaned.to_csv(cleaned_path, index=False)

    enriched, unmatched, tm_summary = enrich_with_transfermarkt(cleaned, cmap)
    enriched_path = OUTPUT_DIR / "enriched_player_dataset.csv"
    unmatched_path = OUTPUT_DIR / "unmatched_transfermarkt_players.csv"
    enriched.to_csv(enriched_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)

    player_df = aggregate_player_dataset(enriched, cmap)
    inspect_dataset(player_df, "final_player_analysis_dataset")

    importance, metrics, shap_df, pipe = run_feature_importance(player_df)
    correlations = correlation_tables(player_df)
    player_df = expected_score_residuals(player_df, pipe)
    outliers = outlier_detection(player_df)
    if "isolation_forest_outlier" not in player_df and "isolation_forest_outlier" in outliers:
        player_df = player_df.merge(outliers[[cmap.player, "isolation_forest_outlier"]], on=cmap.player, how="left")

    rankings = build_rankings(player_df, cmap)

    final_path = OUTPUT_DIR / "final_player_analysis_dataset.csv"
    feature_path = OUTPUT_DIR / "feature_importance.csv"
    rankings_path = OUTPUT_DIR / "player_rankings.csv"
    best_path = OUTPUT_DIR / "best_players.csv"
    over_path = OUTPUT_DIR / "overperformers.csv"
    under_path = OUTPUT_DIR / "underperformers.csv"
    outlier_path = OUTPUT_DIR / "outliers.csv"

    player_df.to_csv(final_path, index=False)
    importance.to_csv(feature_path, index=False)
    if not shap_df.empty:
        shap_df.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)
    rankings["player_rankings"].to_csv(rankings_path, index=False)
    rankings["best_players"].to_csv(best_path, index=False)
    player_df.sort_values("score_residual", ascending=False).head(100).to_csv(over_path, index=False)
    player_df.sort_values("score_residual").head(100).to_csv(under_path, index=False)
    outliers.to_csv(outlier_path, index=False)
    for name, table in rankings.items():
        if name not in {"player_rankings", "best_players"}:
            table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "score_correlations.csv", index=False)

    make_figures(player_df, importance, outliers, cmap)
    summary_path = write_summary(
        source_path, raw, cleaned, enriched, player_df, importance, correlations, rankings, outliers, tm_summary, metrics
    )

    created = [
        cleaned_path,
        enriched_path,
        final_path,
        feature_path,
        rankings_path,
        best_path,
        over_path,
        under_path,
        outlier_path,
        unmatched_path,
        summary_path,
        OUTPUT_DIR / "score_correlations.csv",
    ] + sorted(FIGURE_DIR.glob("*.png"))
    print("\n=== Created output files ===")
    for path in created:
        print(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        raise
