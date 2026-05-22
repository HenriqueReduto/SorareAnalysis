# Sorare Context Dataset

This folder builds a richer analysis dataset by combining:

- Sorare scores from `sorare_outputs/club_player_gw_matched_scores.csv`
- Transfermarkt matchday squads from `data/club_matchday_squads.csv`
- Transfermarkt `games`, `club_games`, `appearances`, and `game_events`

Run from the project root:

```powershell
& "C:\ProgramData\Anaconda3\python.exe" context_dataset\build_context_dataset.py
```

Output:

```text
context_dataset/sorare_context_dataset.csv
context_dataset/context_summary.csv
```

Main added context fields:

- `Opponent Club`
- `Home Away`
- `Team Goals For`
- `Team Goals Against`
- `Result`
- `Team Rest Days`
- `Opponent Season PPG`
- `Opponent Goals For/Game`
- `Opponent Goals Against/Game`
- `Fixture Difficulty`
- `TM Yellow Cards`
- `TM Red Cards`
- `Lineup Status`
- `TM Goals`
- `TM Assists`
- `TM Minutes Played`

Notes:

- Fixture difficulty is based on opponent season performance in Transfermarkt `club_games`.
- The matched Sorare/Transfermarkt player link is name/date based, so ambiguous names should be reviewed.
- Injuries are not included because the public Transfermarkt dataset used here does not expose a reliable injury table.
- `game_events` is very large. To add event-derived cards/substitutions, run:

```powershell
& "C:\ProgramData\Anaconda3\python.exe" context_dataset\build_context_dataset.py --include-events
```
