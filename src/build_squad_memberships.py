"""Build club/player/date matching data from public Transfermarkt datasets.

This script uses dcaribou's Transfermarkt dataset:
https://github.com/dcaribou/transfermarkt-datasets

It creates two useful files:
- data/club_matchday_squads.csv: player was in matchday squad for a club/game date.
- sorare_outputs/club_player_gw_matched_scores.csv: Sorare scores matched to tracked clubs when possible.

The match is intentionally conservative:
- club comes from a manual Sorare slug -> Transfermarkt club name map;
- players are matched by normalised names;
- dates are matched exactly first, then within a small day tolerance.
"""

from __future__ import annotations

import argparse
import io
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
REMOTE_TABLES = {
    "clubs": f"{BASE_URL}/clubs.csv.gz",
    "games": f"{BASE_URL}/games.csv.gz",
    "game_lineups": f"{BASE_URL}/game_lineups.csv.gz",
    "appearances": f"{BASE_URL}/appearances.csv.gz",
    "game_events": f"{BASE_URL}/game_events.csv.gz",
}


def normalise_name(value: object) -> str:
    """Normalise names for cross-source matching."""
    if pd.isna(value):
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of these columns exist: {candidates}. Available: {df.columns.tolist()}")


def _optional_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing column, or None when all candidates are absent."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalise_lineup_status(value: object) -> str:
    """Map Transfermarkt lineup values to Starter/Bench when possible."""
    if pd.isna(value):
        return "Unknown"
    text = normalise_name(value)
    starter_tokens = {
        "starting lineup",
        "starting line up",
        "starter",
        "starting xi",
        "startelf",
        "lineup",
    }
    bench_tokens = {
        "substitutes",
        "substitute",
        "bench",
        "sub",
        "ersatzbank",
    }
    if text in starter_tokens or "starting" in text or "startelf" in text:
        return "Starter"
    if text in bench_tokens or "substitute" in text or "bench" in text:
        return "Bench"
    return str(value)


def read_remote_table(name: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read a Transfermarkt table by URL."""
    response = requests.get(
        REMOTE_TABLES[name],
        headers={"User-Agent": "Mozilla/5.0 sorare-analysis"},
        timeout=120,
    )
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content), compression="gzip", usecols=usecols)


def load_club_map(path: str | Path) -> pd.DataFrame:
    """Load Sorare slug to Transfermarkt club-name mapping."""
    club_map = pd.read_csv(path)
    required = {"sorare_club_slug", "transfermarkt_club_name"}
    missing = required - set(club_map.columns)
    if missing:
        raise ValueError(f"Club map is missing columns: {sorted(missing)}")
    club_map["tm_club_norm"] = club_map["transfermarkt_club_name"].map(normalise_name)
    return club_map


def resolve_tracked_clubs(club_map: pd.DataFrame) -> pd.DataFrame:
    """Match configured club names to Transfermarkt club IDs."""
    clubs = read_remote_table("clubs")
    club_id_col = _first_existing(clubs, ["club_id"])
    club_name_col = _first_existing(clubs, ["name", "club_name"])
    clubs = clubs[[club_id_col, club_name_col]].rename(
        columns={club_id_col: "tm_club_id", club_name_col: "tm_club_name"}
    )
    clubs["tm_club_norm"] = clubs["tm_club_name"].map(normalise_name)

    resolved = club_map.merge(clubs, on="tm_club_norm", how="left")

    # Transfermarkt names can differ slightly from common club names
    # (for example "Benfica" vs "SL Benfica"). Fill unresolved rows with
    # a conservative contains match.
    unresolved_mask = resolved["tm_club_id"].isna()
    if unresolved_mask.any():
        for idx, row in resolved[unresolved_mask].iterrows():
            query = row["tm_club_norm"]
            if not query:
                continue
            candidates = clubs[
                clubs["tm_club_norm"].str.contains(query, regex=False)
                | pd.Series([query in name for name in clubs["tm_club_norm"]], index=clubs.index)
            ].copy()
            if len(candidates) == 1:
                resolved.loc[idx, "tm_club_id"] = candidates.iloc[0]["tm_club_id"]
                resolved.loc[idx, "tm_club_name"] = candidates.iloc[0]["tm_club_name"]

    missing = resolved[resolved["tm_club_id"].isna()]
    if not missing.empty:
        print("Warning: some clubs were not matched to Transfermarkt club IDs:")
        print(missing[["sorare_club_slug", "transfermarkt_club_name"]].to_string(index=False))
    return resolved.dropna(subset=["tm_club_id"]).copy()


def build_matchday_squads(club_map_path: str | Path, start_date: str, output_path: str | Path) -> pd.DataFrame:
    """Build matchday squad rows for tracked clubs."""
    club_map = load_club_map(club_map_path)
    tracked = resolve_tracked_clubs(club_map)
    tracked["tm_club_id"] = tracked["tm_club_id"].astype(str)
    tracked_club_ids = set(tracked["tm_club_id"])

    games = read_remote_table("games")
    game_id_col = _first_existing(games, ["game_id"])
    date_col = _first_existing(games, ["date"])
    games = games[[game_id_col, date_col]].rename(columns={game_id_col: "game_id", date_col: "match_date"})
    games["match_date"] = pd.to_datetime(games["match_date"], utc=True, errors="coerce")
    games = games[games["match_date"] >= pd.to_datetime(start_date, utc=True)].copy()

    try:
        lineups = read_remote_table("game_lineups")
        player_name_col = _first_existing(lineups, ["player_name", "name"])
        player_id_col = _first_existing(lineups, ["player_id"])
        club_id_col = _first_existing(lineups, ["club_id"])
        lineup_game_col = _first_existing(lineups, ["game_id"])
        lineup_status_col = _optional_existing(
            lineups,
            ["type", "lineup_type", "lineup_status", "status", "starting_lineup"],
        )
        keep_cols = [lineup_game_col, club_id_col, player_id_col, player_name_col]
        if lineup_status_col:
            keep_cols.append(lineup_status_col)
        rename_cols = {
            lineup_game_col: "game_id",
            club_id_col: "tm_club_id",
            player_id_col: "tm_player_id",
            player_name_col: "tm_player_name",
        }
        if lineup_status_col:
            rename_cols[lineup_status_col] = "Lineup Status Raw"
        lineups = lineups[keep_cols].rename(columns=rename_cols)
        if "Lineup Status Raw" in lineups.columns:
            lineups["Lineup Status"] = lineups["Lineup Status Raw"].map(normalise_lineup_status)
        else:
            lineups["Lineup Status"] = "In matchday squad"
    except Exception as exc:
        print(f"Could not use game_lineups ({exc}). Falling back to appearances only.")
        appearances = read_remote_table("appearances")
        player_name_col = _first_existing(appearances, ["player_name", "name"])
        player_id_col = _first_existing(appearances, ["player_id"])
        club_id_col = _first_existing(appearances, ["club_id", "player_club_id"])
        app_game_col = _first_existing(appearances, ["game_id"])
        lineups = appearances[[app_game_col, club_id_col, player_id_col, player_name_col]].rename(
            columns={
                app_game_col: "game_id",
                club_id_col: "tm_club_id",
                player_id_col: "tm_player_id",
                player_name_col: "tm_player_name",
            }
        )
        lineups["Lineup Status"] = "Played / starter unknown"

    lineups["tm_club_id"] = lineups["tm_club_id"].astype(str)
    lineups = lineups[lineups["tm_club_id"].isin(tracked_club_ids)].copy()
    squads = lineups.merge(games, on="game_id", how="inner")

    # Add player match stats when available. Transfermarkt appearances usually
    # includes goals/assists/minutes by player and game.
    try:
        appearances = read_remote_table("appearances")
        app_game_col = _first_existing(appearances, ["game_id"])
        app_player_col = _first_existing(appearances, ["player_id"])
        stat_cols: dict[str, str] = {}
        for output_col, candidates in {
            "TM Goals": ["goals", "goals_scored"],
            "TM Assists": ["assists"],
            "TM Minutes Played": ["minutes_played", "minutes"],
        }.items():
            source_col = _optional_existing(appearances, candidates)
            if source_col:
                stat_cols[source_col] = output_col

        if stat_cols:
            appearances_stats = appearances[[app_game_col, app_player_col, *stat_cols.keys()]].rename(
                columns={
                    app_game_col: "game_id",
                    app_player_col: "tm_player_id",
                    **stat_cols,
                }
            )
            appearances_stats["tm_player_id"] = appearances_stats["tm_player_id"].astype(str)
            squads["tm_player_id"] = squads["tm_player_id"].astype(str)
            squads = squads.merge(appearances_stats, on=["game_id", "tm_player_id"], how="left")
        else:
            print("Appearances table did not expose goals/assists/minutes columns.")
    except Exception as exc:
        print(f"Could not add appearance goals/assists/minutes: {exc}")

    # If appearance stats were absent, try deriving goals/assists from game_events.
    try:
        missing_goal_assist = "TM Goals" not in squads.columns or "TM Assists" not in squads.columns
        if missing_goal_assist:
            events = read_remote_table("game_events")
            event_game_col = _first_existing(events, ["game_id"])
            event_player_col = _first_existing(events, ["player_id"])
            event_type_col = _optional_existing(events, ["type", "event_type"])
            event_cols = [event_game_col, event_player_col]
            if event_type_col:
                event_cols.append(event_type_col)
            events = events[event_cols].rename(
                columns={
                    event_game_col: "game_id",
                    event_player_col: "tm_player_id",
                    event_type_col: "event_type" if event_type_col else event_type_col,
                }
            )
            events["tm_player_id"] = events["tm_player_id"].astype(str)
            if "event_type" in events.columns:
                events["event_norm"] = events["event_type"].map(normalise_name)
                event_stats = events.groupby(["game_id", "tm_player_id"]).agg(
                    event_goals=("event_norm", lambda s: s.isin(["goals", "goal"]).sum()),
                    event_assists=("event_norm", lambda s: s.isin(["assists", "assist"]).sum()),
                ).reset_index()
                squads = squads.merge(event_stats, on=["game_id", "tm_player_id"], how="left")
                if "TM Goals" not in squads.columns:
                    squads["TM Goals"] = squads["event_goals"]
                else:
                    squads["TM Goals"] = squads["TM Goals"].fillna(squads["event_goals"])
                if "TM Assists" not in squads.columns:
                    squads["TM Assists"] = squads["event_assists"]
                else:
                    squads["TM Assists"] = squads["TM Assists"].fillna(squads["event_assists"])
                squads = squads.drop(columns=[col for col in ["event_goals", "event_assists"] if col in squads.columns])
    except Exception as exc:
        print(f"Could not add event-derived goals/assists: {exc}")

    for col in ["TM Goals", "TM Assists", "TM Minutes Played"]:
        if col not in squads.columns:
            squads[col] = pd.NA

    squads = squads.merge(
        tracked[["sorare_club_slug", "transfermarkt_club_name", "tm_club_id"]],
        on="tm_club_id",
        how="left",
    )
    squads["tm_player_norm"] = squads["tm_player_name"].map(normalise_name)
    squads = squads.drop_duplicates(
        subset=["sorare_club_slug", "tm_player_id", "match_date"]
    ).sort_values(["sorare_club_slug", "match_date", "tm_player_name"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    squads.to_csv(output, index=False)
    print(f"Saved matchday squads: {output} ({len(squads)} rows)")
    return squads


def match_sorare_scores_to_clubs(
    scores_path: str | Path,
    squads_path: str | Path,
    output_path: str | Path,
    tolerance_days: int = 1,
) -> pd.DataFrame:
    """Match Sorare score rows to tracked clubs by player name and date."""
    scores = pd.read_csv(scores_path)
    squads = pd.read_csv(squads_path)

    scores["Game Date"] = pd.to_datetime(scores["Game Date"], utc=True, errors="coerce")
    squads["match_date"] = pd.to_datetime(squads["match_date"], utc=True, errors="coerce")
    scores["sorare_player_norm"] = scores["Player"].map(normalise_name)

    scores["_score_row_id"] = range(len(scores))
    merged = scores.merge(
        squads[[
            "sorare_club_slug",
            "transfermarkt_club_name",
            "tm_player_id",
            "tm_player_name",
            "tm_player_norm",
            "match_date",
            "Lineup Status",
            "TM Goals",
            "TM Assists",
            "TM Minutes Played",
        ]],
        left_on="sorare_player_norm",
        right_on="tm_player_norm",
        how="left",
    )
    merged["date_diff_days"] = (merged["Game Date"] - merged["match_date"]).abs().dt.days
    candidates = merged[merged["date_diff_days"].le(tolerance_days)].copy()
    candidates = candidates.sort_values(["_score_row_id", "date_diff_days"]).drop_duplicates("_score_row_id")

    match_cols = candidates[[
        "_score_row_id",
        "sorare_club_slug",
        "transfermarkt_club_name",
        "tm_player_id",
        "tm_player_name",
        "match_date",
        "Lineup Status",
        "TM Goals",
        "TM Assists",
        "TM Minutes Played",
        "date_diff_days",
    ]].rename(columns={
        "sorare_club_slug": "Matched Club Slug",
        "transfermarkt_club_name": "Matched Club",
        "match_date": "Matched Match Date",
        "date_diff_days": "Matched Date Diff Days",
    })

    out = scores.merge(match_cols, on="_score_row_id", how="left").drop(columns=["_score_row_id"])
    out["Matched In Dataset Club"] = out["Matched Club Slug"].notna()
    out["Lineup Status"] = out["Lineup Status"].fillna("DNP / not in tracked club matchday squad")
    for col in ["TM Goals", "TM Assists", "TM Minutes Played"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Had Goal"] = out.get("TM Goals", pd.Series(0, index=out.index)).fillna(0).gt(0)
    out["Had Assist"] = out.get("TM Assists", pd.Series(0, index=out.index)).fillna(0).gt(0)
    out["Had Goal Contribution"] = out["Had Goal"] | out["Had Assist"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved matched scores: {output_path} ({len(out)} rows)")
    print(f"Matched rows: {int(out['Matched In Dataset Club'].sum())}")
    print(f"Unmatched rows: {int((~out['Matched In Dataset Club']).sum())}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build club/player/GW matching data from Transfermarkt.")
    parser.add_argument("--scores", default="sorare_outputs/all_players_scores_final.csv")
    parser.add_argument("--club-map", default="config/club_name_map.csv")
    parser.add_argument("--start-date", default="2024-08-01")
    parser.add_argument("--squads-output", default="data/club_matchday_squads.csv")
    parser.add_argument("--matched-output", default="sorare_outputs/club_player_gw_matched_scores.csv")
    parser.add_argument("--tolerance-days", type=int, default=1)
    parser.add_argument("--build-squads-only", action="store_true", help="Only build Transfermarkt squad data.")
    parser.add_argument("--match-only", action="store_true", help="Only match existing squad data to Sorare scores.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_squads_only and args.match_only:
        raise ValueError("Use only one of --build-squads-only or --match-only.")

    if not args.match_only:
        build_matchday_squads(args.club_map, args.start_date, args.squads_output)

    if not args.build_squads_only:
        match_sorare_scores_to_clubs(args.scores, args.squads_output, args.matched_output, args.tolerance_days)


if __name__ == "__main__":
    main()
