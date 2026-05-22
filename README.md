# Sorare Football Score Analysis

This repository extracts Sorare football player scores, prepares Transfermarkt match context, joins both sources, and runs analyses to understand and forecast Sorare score performance.

The project is organized around three main stages:

1. `data_extraction/`: collect source data from Sorare and Transfermarkt.
2. `data_preparation/`: clean, reshape, and combine datasets.
3. `analyses/`: generate summaries, charts, variable-impact tables, and forecast model outputs.

## Quick Start

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

Run the standard Sorare score analysis:

```powershell
python src\main.py --input data_preparation\datasets\sorare\all_players_scores_long.csv
```

Run the Sorare + Transfermarkt matched dataset summary:

```powershell
python src\analyses\analyse_matched_dataset.py
```

Run the score-driver analysis:

```powershell
python src\analyses\analyse_score_drivers.py
```

Run the forecasting models:

```powershell
python src\analyses\forecast_scores.py
```

## Project Structure

```text
SORARE/
  config/
    club_name_map.csv
  data_extraction/
    notebooks/
      sorare_api_extraction.ipynb
    README.md
  data_preparation/
    datasets/
      sorare/
      transfermarkt/
      combined/
    README.md
  analyses/
    matched_dataset/
    notebooks/
    score_variable_impact/
    sorare_lineups/
    sorare_scores/
    README.md
  src/
    data_extraction/
    data_preparation/
    analyses/
    main.py
```

## Data Provenance

The datasets are separated by source so it is clear what contains only Sorare data, only Transfermarkt data, or a join of both.

### Sorare-Only Datasets

Stored in:

```text
data_preparation/datasets/sorare/
```

Important files:

- `all_players_scores_long.csv`: main long-format Sorare score table. This is the preferred input for modelling and statistics.
- `all_players_scores_wide_by_gw.csv`: wide player-by-gameweek table, useful for lineup-style review.

Typical columns in the long file:

- `Club`
- `Club Slug`
- `Player`
- `Player Slug`
- `Position`
- `Score`
- `Game Date`
- `Selected Club`
- `Selected Club Slug`
- `Score Scope`
- `Sorare GW API`
- `Sorare GW`

### Transfermarkt-Only Datasets

Stored in:

```text
data_preparation/datasets/transfermarkt/
```

Important files:

- `club_matchday_squads.csv`: Transfermarkt matchday squad data for tracked clubs.
- `Liga Portugal.xls`: local Transfermarkt-related source workbook.

The matchday squad table contains Transfermarkt identifiers, player names, club IDs, match dates, lineup status, and available appearance stats such as goals, assists, and minutes.

### Combined Sorare + Transfermarkt Datasets

Stored in:

```text
data_preparation/datasets/combined/
```

Important files:

- `club_player_gw_matched_scores.csv`: Sorare score rows matched to Transfermarkt club/player/matchday data.
- `sorare_transfermarkt_context.csv`: richer modelling dataset with Sorare scores plus Transfermarkt fixture, lineup, opponent, result, and appearance context.
- `context_summary.csv`: compact summary of the context dataset.

## Data Flow

The intended pipeline is:

```text
Sorare API extraction notebook
  -> data_preparation/datasets/sorare/all_players_scores_long.csv
  -> src/data_extraction/build_squad_memberships.py
  -> data_preparation/datasets/transfermarkt/club_matchday_squads.csv
  -> data_preparation/datasets/combined/club_player_gw_matched_scores.csv
  -> src/data_preparation/build_context_dataset.py
  -> data_preparation/datasets/combined/sorare_transfermarkt_context.csv
  -> analysis scripts in src/analyses/
  -> outputs in analyses/
```

## Data Extraction

### Sorare Extraction Notebook

The Sorare API extraction notebook is:

```text
data_extraction/notebooks/sorare_api_extraction.ipynb
```

It is responsible for collecting player score history from Sorare and writing prepared Sorare-only files under:

```text
data_preparation/datasets/sorare/
```

The long-format output is the most important one:

```text
data_preparation/datasets/sorare/all_players_scores_long.csv
```

### Transfermarkt Squad Builder

The Transfermarkt builder uses the public `dcaribou/transfermarkt-datasets` CSV files and the local club mapping:

```text
config/club_name_map.csv
```

Run:

```powershell
python src\data_extraction\build_squad_memberships.py --start-date 2024-08-01
```

Default input:

```text
data_preparation/datasets/sorare/all_players_scores_long.csv
```

Default outputs:

```text
data_preparation/datasets/transfermarkt/club_matchday_squads.csv
data_preparation/datasets/combined/club_player_gw_matched_scores.csv
```

The matching is intentionally conservative:

- clubs are linked through `config/club_name_map.csv`;
- player names are normalized before matching;
- match dates are matched exactly first, then with a small tolerance;
- unmatched Sorare rows are kept, with `Matched In Dataset Club = False`.

Useful optional arguments:

```powershell
python src\data_extraction\build_squad_memberships.py --build-squads-only
python src\data_extraction\build_squad_memberships.py --match-only
python src\data_extraction\build_squad_memberships.py --tolerance-days 2
```

## Data Preparation

### Build Context Dataset

The context builder creates the main combined modelling dataset:

```powershell
python src\data_preparation\build_context_dataset.py
```

It reads:

```text
data_preparation/datasets/combined/club_player_gw_matched_scores.csv
data_preparation/datasets/transfermarkt/club_matchday_squads.csv
```

It writes:

```text
data_preparation/datasets/combined/sorare_transfermarkt_context.csv
data_preparation/datasets/combined/context_summary.csv
```

The context dataset adds fields such as:

- `Opponent Club`
- `Home Away`
- `Team Goals For`
- `Team Goals Against`
- `Result`
- `Team Rest Days`
- `Team Position`
- `Opponent Position`
- `Team Formation`
- `Opponent Season PPG`
- `Fixture Difficulty`
- `TM Yellow Cards`
- `TM Red Cards`
- `TM Appearance Minutes`
- `Context Match Available`

By default, the script skips the very large Transfermarkt `game_events` table. To include event-derived cards and substitutions:

```powershell
python src\data_preparation\build_context_dataset.py --include-events
```

Downloaded Transfermarkt source tables are cached under:

```text
data_preparation/cache/transfermarkt/
```

That cache is ignored by git because it can become large and can be recreated.

## Analyses

### Standard Sorare Score Analysis

Run:

```powershell
python src\main.py --input data_preparation\datasets\sorare\all_players_scores_long.csv
```

Default output folder:

```text
analyses/sorare_scores/
```

This pipeline:

1. loads the Sorare score dataset;
2. inspects shape, columns, missing values, duplicates, and inferred column types;
3. cleans score rows;
4. creates historical score features;
5. builds player rankings and value rankings;
6. saves charts;
7. trains a baseline next-score model.

Main outputs:

- `cleaned_sorare_scores.csv`
- `engineered_sorare_scores.csv`
- `player_rankings.csv`
- `value_rankings.csv`
- `prediction_results.csv`
- `analysis_summary.md`
- `model_metrics.md`
- charts under `analyses/sorare_scores/charts/`

### Matched Dataset Analysis

Run:

```powershell
python src\analyses\analyse_matched_dataset.py
```

Input:

```text
data_preparation/datasets/combined/club_player_gw_matched_scores.csv
```

Outputs:

```text
analyses/matched_dataset/
```

Generated tables:

- `lineup_summary.csv`: average score by starter, bench, and DNP/unmatched status.
- `player_summary.csv`: player-level matched performance summary.
- `goal_contribution_summary.csv`: score effect of goals or assists.
- `club_summary.csv`: club-level matched score summary.

### Score Variable Impact Analysis

Run:

```powershell
python src\analyses\analyse_score_drivers.py
```

Input:

```text
data_preparation/datasets/combined/sorare_transfermarkt_context.csv
```

Outputs:

```text
analyses/score_variable_impact/
```

Generated files:

- `analysis_summary.md`
- `model_permutation_importance.csv`
- `linear_grouped_coefficients.csv`
- `numeric_correlations.csv`
- `categorical_effects.csv`

This analysis tests which variables explain Sorare score variation. It includes match outcome and usage variables such as goals, assists, minutes, cards, and lineup status, so it is useful for explanation but not always valid for pre-match forecasting.

### Forecast Models

Run:

```powershell
python src\analyses\forecast_scores.py
```

Input:

```text
data_preparation/datasets/combined/sorare_transfermarkt_context.csv
```

Outputs:

```text
analyses/score_variable_impact/
```

Generated files:

- `forecast_model_summary.md`
- `pre_match_forecast_test_predictions.csv`
- `pre_match_forecast_feature_importance.csv`
- `pre_match_forecast_prediction_buckets.csv`
- `lineup_aware_forecast_test_predictions.csv`
- `lineup_aware_forecast_feature_importance.csv`
- `lineup_aware_forecast_prediction_buckets.csv`

The forecast script trains two models:

- `pre_match_forecast`: excludes match outcomes, goals, assists, cards, result, and minutes. This is the fairer model for lineup-building decisions.
- `lineup_aware_forecast`: includes lineup status. Use this only when expected or official starter/bench status is known.

The model uses a temporal split: older matches train the model, newer matches test it.

## Important Files After Reorganization

If your IDE still has old paths open, use the new paths below:

| Old path | New path |
| --- | --- |
| `context_dataset/score_variable_impact_analysis/forecast_model_summary.md` | `analyses/score_variable_impact/forecast_model_summary.md` |
| `context_dataset/score_variable_impact_analysis/forecast_scores.py` | `src/analyses/forecast_scores.py` |
| `context_dataset/score_variable_impact_analysis/linear_grouped_coefficients.csv` | `analyses/score_variable_impact/linear_grouped_coefficients.csv` |
| `context_dataset/score_variable_impact_analysis/categorical_effects.csv` | `analyses/score_variable_impact/categorical_effects.csv` |
| `context_dataset/score_variable_impact_analysis/model_permutation_importance.csv` | `analyses/score_variable_impact/model_permutation_importance.csv` |
| `sorare_outputs/all_players_scores_final.csv` | `data_preparation/datasets/sorare/all_players_scores_long.csv` |
| `sorare_outputs/club_player_gw_matched_scores.csv` | `data_preparation/datasets/combined/club_player_gw_matched_scores.csv` |
| `data/club_matchday_squads.csv` | `data_preparation/datasets/transfermarkt/club_matchday_squads.csv` |

## Source Code Map

Data extraction:

- `src/data_extraction/load_data.py`: loads local CSV or Excel files.
- `src/data_extraction/build_squad_memberships.py`: downloads Transfermarkt data, builds matchday squads, and matches Sorare scores to clubs.

Data preparation:

- `src/data_preparation/clean_data.py`: normalizes and cleans Sorare score data.
- `src/data_preparation/feature_engineering.py`: builds historical score features without using future matches.
- `src/data_preparation/build_context_dataset.py`: creates the rich Sorare + Transfermarkt context dataset.
- `src/data_preparation/squad_matching.py`: helper logic for squad-style matching.

Analyses:

- `src/main.py`: standard Sorare score analysis runner.
- `src/analyses/analyse_scores.py`: ranking and summary tables.
- `src/analyses/analyse_matched_dataset.py`: summaries for matched Sorare + Transfermarkt data.
- `src/analyses/analyse_score_drivers.py`: explanatory variable-impact modelling.
- `src/analyses/forecast_scores.py`: leakage-aware forecasting models.
- `src/analyses/model_predictions.py`: baseline next-score prediction model.
- `src/analyses/visualise.py`: chart generation.
- `src/analyses/inspect_dataset.py`: dataset inspection helper.

## Methodology Notes

- Missing Sorare scores are not treated as zero during cleaning or feature engineering.
- Wide Sorare tables are mainly for lineup-style review. Use the long table for modelling and statistics.
- Rolling features use prior matches only to avoid lookahead bias.
- The combined context dataset is linked by normalized player name and match date, so ambiguous names should be reviewed.
- Transfermarkt goals, assists, minutes, cards, team goals, and match result are post-match signals. They explain score but should not be used as pre-match prediction inputs.
- The pre-match forecast model is designed to avoid obvious match-outcome leakage.
- Forecasts should be treated as ranking signals, not exact score predictions.

## Refreshing the Whole Project

Use this sequence when starting from a fresh Sorare extraction:

```powershell
# 1. Run or update the Sorare extraction notebook:
data_extraction\notebooks\sorare_api_extraction.ipynb

# 2. Rebuild Transfermarkt matchday squads and matched scores:
python src\data_extraction\build_squad_memberships.py --start-date 2024-08-01

# 3. Rebuild the rich context dataset:
python src\data_preparation\build_context_dataset.py

# 4. Rebuild analysis outputs:
python src\main.py --input data_preparation\datasets\sorare\all_players_scores_long.csv
python src\analyses\analyse_matched_dataset.py
python src\analyses\analyse_score_drivers.py
python src\analyses\forecast_scores.py
```

## Troubleshooting

If a script cannot find a file, check whether the dataset exists in the correct provenance folder:

- Sorare-only: `data_preparation/datasets/sorare/`
- Transfermarkt-only: `data_preparation/datasets/transfermarkt/`
- Combined: `data_preparation/datasets/combined/`

If Transfermarkt downloads fail, rerun the command later or remove the cache folder:

```powershell
Remove-Item -Recurse -Force data_preparation\cache\transfermarkt
```

If the console has trouble printing names with accents, use the Anaconda Python referenced in the scripts:

```powershell
& "C:\ProgramData\Anaconda3\python.exe" src\analyses\forecast_scores.py
```

The scripts also configure UTF-8 output where needed, but terminal encoding can still vary by environment.
