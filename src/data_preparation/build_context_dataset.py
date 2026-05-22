"""Build the richest possible Sorare + Transfermarkt context dataset."""

from __future__ import annotations

import io
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_extraction.build_squad_memberships import normalise_name  # noqa: E402


BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
REMOTE_TABLES = {
    "games": f"{BASE_URL}/games.csv.gz",
    "club_games": f"{BASE_URL}/club_games.csv.gz",
    "appearances": f"{BASE_URL}/appearances.csv.gz",
    "game_events": f"{BASE_URL}/game_events.csv.gz",
}

DATASETS_DIR = PROJECT_ROOT / "data_preparation" / "datasets"
MATCHED_SCORES = DATASETS_DIR / "combined" / "club_player_gw_matched_scores.csv"
MATCHDAY_SQUADS = DATASETS_DIR / "transfermarkt" / "club_matchday_squads.csv"
OUTPUT_FILE = DATASETS_DIR / "combined" / "sorare_transfermarkt_context.csv"
SUMMARY_FILE = DATASETS_DIR / "combined" / "context_summary.csv"
CACHE_DIR = PROJECT_ROOT / "data_preparation" / "cache" / "transfermarkt"


def read_remote_table(name: str) -> pd.DataFrame:
    """Read a public Transfermarkt CSV table."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{name}.csv"
    if cache_file.exists():
        print(f"Loading cached {name}...")
        return pd.read_csv(cache_file, low_memory=False)

    print(f"Downloading {name}...")
    response = requests.get(
        REMOTE_TABLES[name],
        headers={"User-Agent": "Mozilla/5.0 sorare-context-builder"},
        timeout=180,
    )
    response.raise_for_status()
    df = pd.read_csv(io.BytesIO(response.content), compression="gzip", low_memory=False)
    df.to_csv(cache_file, index=False)
    return df


def result_from_goals(goals_for: float, goals_against: float) -> str:
    """Convert goals for/against to W/D/L."""
    if pd.isna(goals_for) or pd.isna(goals_against):
        return np.nan
    if goals_for > goals_against:
        return "W"
    if goals_for == goals_against:
        return "D"
    return "L"


def points_from_result(result: str) -> int:
    """Return league-style points from result."""
    return {"W": 3, "D": 1, "L": 0}.get(result, 0)


def build_game_context() -> pd.DataFrame:
    """Build club-game context, opponent strength, and rest days."""
    print("Building game and fixture context...")
    games = read_remote_table("games")
    club_games = read_remote_table("club_games")

    games["date"] = pd.to_datetime(games["date"], utc=True, errors="coerce")
    game_cols = [
        "game_id",
        "competition_id",
        "season",
        "round",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_name",
        "away_club_name",
        "home_club_formation",
        "away_club_formation",
    ]
    games = games[[col for col in game_cols if col in games.columns]]

    context = club_games.merge(games, on="game_id", how="left")
    context["club_id"] = context["club_id"].astype(str)
    context["opponent_id"] = context["opponent_id"].astype(str)
    context["Result"] = context.apply(
        lambda r: result_from_goals(r.get("own_goals"), r.get("opponent_goals")),
        axis=1,
    )
    context["Team Points"] = context["Result"].map(points_from_result)
    context["Home Away"] = context["hosting"]
    context["Team Goals For"] = context["own_goals"]
    context["Team Goals Against"] = context["opponent_goals"]
    context["Opponent Position"] = context["opponent_position"]
    context["Team Position"] = context["own_position"]

    context["Team Formation"] = np.where(
        context["Home Away"].eq("Home"),
        context.get("home_club_formation"),
        context.get("away_club_formation"),
    )
    context["Opponent Club"] = np.where(
        context["Home Away"].eq("Home"),
        context.get("away_club_name"),
        context.get("home_club_name"),
    )

    context = context.sort_values(["club_id", "date"])
    context["Team Rest Days"] = context.groupby("club_id")["date"].diff().dt.days

    strength = (
        context.groupby(["club_id", "season"], dropna=False)
        .agg(
            opponent_games=("game_id", "count"),
            opponent_ppg=("Team Points", "mean"),
            opponent_goals_for_pg=("Team Goals For", "mean"),
            opponent_goals_against_pg=("Team Goals Against", "mean"),
            opponent_win_rate=("Result", lambda s: (s == "W").mean()),
        )
        .reset_index()
        .rename(columns={"club_id": "opponent_id"})
    )
    context = context.merge(strength, on=["opponent_id", "season"], how="left")

    context["Fixture Difficulty Score"] = context["opponent_ppg"]
    valid = context["Fixture Difficulty Score"].notna()
    if valid.any():
        context.loc[valid, "Fixture Difficulty"] = pd.qcut(
            context.loc[valid, "Fixture Difficulty Score"].rank(method="first"),
            q=5,
            labels=[1, 2, 3, 4, 5],
        ).astype("Int64")
    else:
        context["Fixture Difficulty"] = pd.NA

    return context[[
        "game_id",
        "club_id",
        "opponent_id",
        "competition_id",
        "season",
        "round",
        "date",
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
    ]]


def build_player_event_context() -> pd.DataFrame:
    """Build player-level card and substitution event counts by game."""
    print("Building event context...")
    events = read_remote_table("game_events")
    if events.empty:
        return pd.DataFrame(columns=["game_id", "tm_player_id"])

    events["event_norm"] = events["type"].map(normalise_name)
    events["description_norm"] = events["description"].fillna("").map(normalise_name)
    events["tm_player_id"] = events["player_id"].astype("Int64").astype(str)

    event_context = (
        events.dropna(subset=["player_id"])
        .groupby(["game_id", "tm_player_id"], dropna=False)
        .agg(
            event_yellow_cards=("description_norm", lambda s: s.str.contains("yellow card", regex=False).sum()),
            event_red_cards=("description_norm", lambda s: s.str.contains("red card", regex=False).sum()),
            event_substitutions=("event_norm", lambda s: s.eq("substitutions").sum()),
        )
        .reset_index()
        .rename(columns={
            "event_yellow_cards": "Event Yellow Cards",
            "event_red_cards": "Event Red Cards",
            "event_substitutions": "Event Substitutions",
        })
    )
    return event_context


def build_appearance_context() -> pd.DataFrame:
    """Build player appearance context from Transfermarkt appearances."""
    print("Building appearance context...")
    apps = read_remote_table("appearances")
    apps["tm_player_id"] = apps["player_id"].astype("Int64").astype(str)
    apps = apps.rename(columns={
        "yellow_cards": "TM Yellow Cards",
        "red_cards": "TM Red Cards",
        "minutes_played": "TM Appearance Minutes",
    })
    keep = [
        "game_id",
        "tm_player_id",
        "TM Yellow Cards",
        "TM Red Cards",
        "TM Appearance Minutes",
    ]
    return apps[[col for col in keep if col in apps.columns]].drop_duplicates(["game_id", "tm_player_id"])


def build_context_dataset(include_events: bool = False) -> pd.DataFrame:
    """Build and save the final context dataset."""
    print("Loading matched Sorare scores and matchday squads...")
    matched = pd.read_csv(MATCHED_SCORES)
    squads = pd.read_csv(MATCHDAY_SQUADS)

    matched["Game Date"] = pd.to_datetime(matched["Game Date"], utc=True, errors="coerce")
    matched["Matched Match Date"] = pd.to_datetime(matched["Matched Match Date"], utc=True, errors="coerce")
    squads["match_date"] = pd.to_datetime(squads["match_date"], utc=True, errors="coerce")

    matched["tm_player_id"] = pd.to_numeric(matched["tm_player_id"], errors="coerce").astype("Int64").astype(str)
    squads["tm_player_id"] = pd.to_numeric(squads["tm_player_id"], errors="coerce").astype("Int64").astype(str)
    squads["tm_club_id"] = squads["tm_club_id"].astype(str)

    join_cols = ["sorare_club_slug", "tm_player_id", "match_date"]
    squad_lookup = squads[[
        "sorare_club_slug",
        "tm_player_id",
        "match_date",
        "game_id",
        "tm_club_id",
    ]].drop_duplicates(join_cols)

    enriched = matched.merge(
        squad_lookup,
        left_on=["Matched Club Slug", "tm_player_id", "Matched Match Date"],
        right_on=join_cols,
        how="left",
    )

    game_context = build_game_context()
    print("Merging fixture context...")
    enriched = enriched.merge(
        game_context,
        left_on=["game_id", "tm_club_id"],
        right_on=["game_id", "club_id"],
        how="left",
    )

    appearance_context = build_appearance_context()
    print("Merging appearance context...")
    enriched = enriched.merge(appearance_context, on=["game_id", "tm_player_id"], how="left")

    if include_events:
        event_context = build_player_event_context()
        print("Merging event context...")
        enriched = enriched.merge(event_context, on=["game_id", "tm_player_id"], how="left")
    else:
        print("Skipping game_events context by default because it is very large. Use --include-events if needed.")
        enriched["Event Yellow Cards"] = pd.NA
        enriched["Event Red Cards"] = pd.NA
        enriched["Event Substitutions"] = pd.NA

    for col in ["TM Yellow Cards", "TM Red Cards", "Event Yellow Cards", "Event Red Cards", "Event Substitutions"]:
        if col in enriched.columns:
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(0)

    enriched["Context Match Available"] = enriched["game_id"].notna()
    enriched["Fixture Difficulty Label"] = enriched["Fixture Difficulty"].map({
        1: "Very easy",
        2: "Easy",
        3: "Medium",
        4: "Hard",
        5: "Very hard",
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_FILE, index=False)

    summary = (
        enriched.groupby(["Lineup Status", "Fixture Difficulty Label"], dropna=False)
        .agg(
            rows=("Score", "count"),
            avg_score=("Score", "mean"),
            median_score=("Score", "median"),
            goals=("TM Goals", "sum"),
            assists=("TM Assists", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(SUMMARY_FILE, index=False)
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rich Sorare + Transfermarkt context dataset.")
    parser.add_argument("--include-events", action="store_true", help="Also process the very large game_events table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_context_dataset(include_events=args.include_events)
    print(f"Saved context dataset: {OUTPUT_FILE} ({len(dataset)} rows)")
    print(f"Saved context summary: {SUMMARY_FILE}")
    print("Context match rows:", int(dataset["Context Match Available"].sum()))
    print("Columns:", len(dataset.columns))


if __name__ == "__main__":
    main()
