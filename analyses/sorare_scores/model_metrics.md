# Model Metrics

- MAE: 14.916109915137103
- RMSE: 21.381689522369943
- R2: 0.34652420064502953
- feature_importance: {'rolling_5_avg_score': np.float64(0.46702262827932395), 'rolling_15_avg_score': np.float64(0.12305018957870606), 'median_score': np.float64(0.10014197299471367), 'score_volatility': np.float64(0.06022953375623951), 'score_floor': np.float64(1.623963773349115e-06), 'score_ceiling': np.float64(0.0355746141487221), 'consistency_index': np.float64(0.09771259480443387), 'form_delta': np.float64(0.11626684247408754), 'dnp_rate': np.float64(0.0), 'minutes_played_rate': np.float64(0.0), 'score_per_eur': np.float64(0.0)}
- limitations: Baseline model only uses historical score features and does not include injuries, lineups, opponent, home/away, or transfer context.
