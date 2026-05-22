# Score Variable Impact Analysis

- Dataset rows used: 32,638
- Variables tested: 37
- Random forest test R2: 0.502
- Random forest test MAE: 14.59
- Random forest test RMSE: 19.76

## Most Important Variables

- Lineup Status: R2 drop 0.7465 (std 0.0183)
- Position: R2 drop 0.0702 (std 0.0035)
- Had Goal Contribution: R2 drop 0.0671 (std 0.0013)
- TM Appearance Minutes: R2 drop 0.0666 (std 0.0026)
- TM Minutes Played: R2 drop 0.0568 (std 0.0023)
- Team Goals Against: R2 drop 0.0252 (std 0.0008)
- Selected Club: R2 drop 0.0099 (std 0.0021)
- Club: R2 drop 0.0098 (std 0.0022)
- Matched Club: R2 drop 0.0035 (std 0.0004)
- Opponent Position: R2 drop 0.0020 (std 0.0004)
- Matched In Dataset Club: R2 drop 0.0012 (std 0.0008)
- Team Position: R2 drop 0.0006 (std 0.0001)
- Opponent Club: R2 drop 0.0005 (std 0.0006)
- TM Goals: R2 drop 0.0005 (std 0.0000)
- TM Yellow Cards: R2 drop 0.0005 (std 0.0000)

## Strongest Numeric Relationships

- TM Goals: Spearman 0.414, Pearson 0.450
- TM Appearance Minutes: Spearman 0.403, Pearson 0.379
- TM Minutes Played: Spearman 0.403, Pearson 0.379
- Context Match Available: Spearman 0.376, Pearson 0.367
- Matched In Dataset Club: Spearman 0.376, Pearson 0.367
- Had Goal Contribution: Spearman 0.367, Pearson 0.406
- TM Assists: Spearman 0.358, Pearson 0.401
- Had Goal: Spearman 0.282, Pearson 0.314
- Had Assist: Spearman 0.254, Pearson 0.286
- Team Points: Spearman 0.178, Pearson 0.169

## Strongest Category Score Gaps

- Lineup Status = Starter: +26.58 points vs average
- Opponent Club = St. Johnstone FC: +23.73 points vs average
- Team Formation = 3-5-2 Attacking: +18.92 points vs average
- Result = W: +15.32 points vs average
- Fixture Difficulty Label = Very easy: +14.77 points vs average
- Matched Club = The Celtic Football Club: +14.55 points vs average
- Home Away = Home: +12.59 points vs average
- Position = Goalkeeper: -11.71 points vs average
- Club = FC Red Bull Salzburg: -4.00 points vs average
- Selected Club = FC Red Bull Salzburg: -4.00 points vs average

## Reading Notes

- `TM Goals`, `TM Assists`, cards, minutes, and lineup status are match outcomes or usage signals, so they explain score strongly but are not all pre-match predictors.
- `permutation_importance` is usually the best table for overall impact because it captures non-linear effects and interactions.
- `linear_grouped_coefficients.csv` is useful for direction, but categorical variables are one-hot encoded and grouped back to their source variable.