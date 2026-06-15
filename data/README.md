# data/ — downloaded Kaggle training-run artifacts

Drop the outputs of `examples/kaggle_training.ipynb` (downloaded from the Kaggle
notebook's `/kaggle/working`) here, then run `examples/analyze_kaggle_run.ipynb`
to analyze them.

Expected files:

| File | Produced by | Used for |
|---|---|---|
| `manifest.json` | training notebook | which window won + run metadata |
| `model_comparison.csv` | training notebook | headline accuracy/skill/coverage per window |
| `backtest_scores.csv` | training notebook | per-fold, per-horizon metrics |
| `model_10yr.joblib`, `model_15yr.joblib`, `model_full.joblib` | `LightGBMForecaster.save()` | feature importances, recomputed diagnostics |
| `daily_panel_full.parquet` | training notebook | held-out residual / seasonal / per-asset analysis |

These artifacts are git-ignored (binary/large); only this README is committed.
