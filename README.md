# Sorare Player Score Analysis

Python project for extracting, inspecting, cleaning, analysing, visualising, and modelling Sorare football player scores.

The project supports two data paths:

- Local file input: CSV or Excel.
- Sorare API scaffolding: placeholder GraphQL helpers in `src/load_data.py`.

The current notebook extraction writes:

```text
sorare_outputs/all_players_scores_final.csv
sorare_outputs/all_players_scores_wide_final.csv
```

## Club x Player x GW Matching

To assign a Sorare score to a tracked club only when the player was actually in that club's matchday squad, use the Transfermarkt helper:

```bash
python src/build_squad_memberships.py --scores sorare_outputs/all_players_scores_final.csv --start-date 2024-08-01
```

It uses the public `dcaribou/transfermarkt-datasets` CSV files and the local mapping:

```text
config/club_name_map.csv
```

Outputs:

```text
data/club_matchday_squads.csv
sorare_outputs/club_player_gw_matched_scores.csv
```

Rows that do not match a tracked club/player/date keep `Matched Club` as null. When the Transfermarkt `game_lineups` table exposes lineup type, the matched output also includes `Lineup Status`:

- `Starter`
- `Bench`
- `DNP / not in tracked club matchday squad`

When Transfermarkt appearances/events expose scoring data, the matched output also includes:

- `TM Goals`
- `TM Assists`
- `TM Minutes Played`
- `Had Goal`
- `Had Assist`
- `Had Goal Contribution`

Run the full project from a local file:

```bash
python src/main.py --input sorare_outputs/all_players_scores_final.csv
```

Or with your own CSV:

```bash
python src/main.py --input data/sorare_player_scores.csv
```

## Outputs

Generated files are written to `outputs/`:

- `cleaned_sorare_scores.csv`
- `engineered_sorare_scores.csv`
- `player_rankings.csv`
- `value_rankings.csv`
- `prediction_results.csv`
- `analysis_summary.md`
- charts under `outputs/charts/`

## Sorare API Notes

Sorare's public API is GraphQL at:

```text
https://api.sorare.com/graphql
```

Authenticated access may require a JWT and headers documented by Sorare. Copy `config.example.env` to `.env` and fill values if needed.

The project keeps API code separate from CSV/Excel analysis logic. The notebook is currently the practical extractor; the `src/load_data.py` API functions are safe scaffolding for future expansion.

## Important Methodology Notes

- Missing score is not treated as zero during cleaning or feature engineering.
- Notebook wide tables may fill missing scores for lineup-style views; use the long CSV for modelling and statistics.
- Rolling features use prior matches only to avoid lookahead bias.
- Whole-history scores are player-history scores from players found in selected/current squad clubs, not necessarily scores only at that club.
