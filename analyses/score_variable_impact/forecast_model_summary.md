# Forecast Model Summary

This uses a temporal split: older matches train the model, newer matches test it.

## Pre Match Forecast

- Train rows: 19,015
- Test rows: 12,749
- Model MAE: 13.72
- Model RMSE: 19.17
- Model R2: 0.510
- Player-average baseline MAE: 16.75
- Global-average baseline MAE: 25.06

Top forecast features:

- player_last_score: 0.1495
- player_rolling_3_score: 0.1391
- player_rolling_5_score: 0.1184
- player_rolling_10_score: 0.1044
- player_avg_score_before: 0.0897
- player_median_score_before: 0.0655
- player_zero_rate_before: 0.0655
- player_std_score_before: 0.0404
- Home Away: 0.0184
- Team Formation: 0.0164
- Matched In Dataset Club: 0.0152
- Opponent Club: 0.0149

## Lineup Aware Forecast

- Train rows: 19,015
- Test rows: 12,749
- Model MAE: 12.88
- Model RMSE: 17.61
- Model R2: 0.587
- Player-average baseline MAE: 16.75
- Global-average baseline MAE: 25.06

Top forecast features:

- Lineup Status: 0.1677
- player_last_score: 0.1194
- player_rolling_3_score: 0.1165
- player_rolling_5_score: 0.0955
- player_rolling_10_score: 0.0949
- player_zero_rate_before: 0.0732
- player_avg_score_before: 0.0714
- player_median_score_before: 0.0521
- player_std_score_before: 0.0302
- Fixture Difficulty Label: 0.0160
- Matched Club: 0.0140
- position_avg_score_before: 0.0125

## Reliability Notes

- The pre-match model is the fair lineup-building model because it excludes match outcomes, minutes, goals, assists, result, and cards.
- The lineup-aware model can be used only when you already know expected or official starter status.
- Sorare scores are noisy, so use predictions as ranking signals, not exact point forecasts.
- The model should be retrained whenever the dataset is refreshed.