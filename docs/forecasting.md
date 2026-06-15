# Forecasting

climagrid's forecasting module predicts each asset's **daily stress features N
days ahead**, with prediction intervals, so a maintenance team gets lead time
to schedule inspections before a stress peak.

```{important}
These are forecasts of environmental **stress**, not equipment failure.
climagrid does not predict failures (see [Validation Notes](validation_notes.md)).
A forecast says "this transformer's heat-aging stress is likely to climb over
the next week"; it does not say the transformer will fail.
```

## Install

Forecasting needs the optional `[ml]` extra (LightGBM, scikit-learn, pyarrow):

```bash
pip install "climagrid[ml]"
```

## Quick start

```python
import climagrid
from datetime import datetime, timezone

forecast = climagrid.forecast(
    "my_assets.csv",                      # asset_id, lat, lon
    history_start=datetime(2010, 1, 1, tzinfo=timezone.utc),
    history_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
)
print(forecast.head())
```

The result is long form, one row per `(asset_id, origin_date, target,
horizon_day)`:

| Column | Meaning |
|---|---|
| `asset_id` | Asset identifier |
| `origin_date` | The day the forecast is issued from |
| `forecast_date` | `origin_date + horizon_day` |
| `horizon_day` | Lead time in days (1..H) |
| `target` | Stress feature being forecast |
| `p10` / `p50` / `p90` | 10th / 50th / 90th percentile forecast |

`p50` is the point forecast; `p10`-`p90` is an 80% prediction interval. The
quantiles are guaranteed non-decreasing (`p10 <= p50 <= p90`).

## Configuration

Everything is controlled by `ForecastConfig`:

```python
from climagrid import ForecastConfig

config = ForecastConfig(
    targets=["feat_thermal_aging_factor"],  # which stress features to forecast
    horizon_days=7,                         # forecast up to 7 days ahead
    history_years=15,                       # default training history length
    quantiles=[0.1, 0.5, 0.9],              # prediction-interval quantiles
    model="lightgbm",                       # or "persistence" / "climatology"
)

forecast = climagrid.forecast("my_assets.csv", config=config)
```

The default target is `feat_thermal_aging_factor`, a per-row Arrhenius function
of temperature where a learned model has the most to add over naive baselines.
Forecasting more targets trains more models (one per target, horizon, and
quantile).

## How it works

1. **Panel build** ([`build_training_panel`](api/forecast.md)): fetches hourly
   climagrid features per asset (streaming one asset at a time to bound memory)
   and aggregates them to one value per day (default: daily max).
2. **Supervised frame** ([`build_supervised_frame`](api/forecast.md)): builds
   strictly backward-looking predictors, autoregressive lags, trailing rolling
   mean/std, day-of-year harmonics and static location, plus the shifted target
   columns `y_h1..y_hH`.
3. **Model**: one LightGBM quantile regressor per `(horizon, quantile)` using
   the direct multi-horizon strategy. A single global model is pooled across
   assets so it generalizes to asset locations it has not seen.
4. **Forecast**: predicts forward from each asset's most recent day.

All predictors are causal: rows are sorted by `(asset_id, date)` first, lags
look only backward, and rolling windows exclude the origin day.

## Honest evaluation: baselines and backtesting

A forecast is only worth trusting if it beats simple baselines out of sample.
The module ships both:

- **Persistence**: tomorrow looks like today.
- **Climatology**: the historical day-of-year average.

`evaluate` runs a rolling-origin backtest with an embargo gap (so a training
target window never overlaps a test predictor window) and reports skill scores
against both baselines, plus interval calibration:

```python
from climagrid.forecasting import evaluate
from climagrid.forecasting.dataset import build_training_panel

panel = build_training_panel("my_assets.csv", start, end, config)
scores = evaluate(panel, config, n_splits=3, test_size_days=90)
print(scores[["target", "horizon_day", "skill_vs_persistence", "interval_coverage"]])
```

`skill_vs_persistence = 1 - MSE_model / MSE_persistence`: positive means the
model beats persistence, zero means it ties, negative means it loses. Because
stress features are smooth and autocorrelated, persistence is a strong baseline
at short horizons; expect the clearest gains at medium range.

### How much history to use

More history strengthens the seasonal signal but the oldest years reflect a
slightly different climate. Rather than guess, `history_ablation` measures it:

```python
from climagrid.forecasting.backtest import history_ablation

ablation = history_ablation(panel, config, windows_years=[10, 15, 25])
```

Pick the history length on measured skill and calibration, not assumption.

## Calibrated prediction intervals

The raw quantile intervals tend to be slightly narrow (in one 33-substation run
the 80 percent interval covered about 0.74 of outcomes against the 0.80 target).
Setting `calibrate_intervals=True` applies conformalized quantile regression: a
held-out calibration window is used to widen the interval so its coverage
matches the nominal level.

```python
config = ForecastConfig(
    calibrate_intervals=True,
    calibration_method="mondrian",  # "constant" | "normalized" | "mondrian"
    calibration_days=365,           # hold out a full year, see note below
)
forecast = climagrid.forecast("my_assets.csv", config=config)
```

In that run calibration lifted overall coverage to about 0.78, close to target.
Three methods are available, trading off marginal vs per-season coverage:

- `"constant"`: a single additive width per horizon. Good marginal coverage.
- `"normalized"`: scales the width by the model's own interval width, so it
  adapts to local uncertainty. Marginally the most even overall, but the
  high-stress summer months stay under-covered (about 0.72 in that run).
- `"mondrian"`: a separate width per meteorological season (keyed on the
  forecast date). Brings **summer** coverage up to target (about 0.78), which
  matters most for a grid-stress tool since thermal aging is exponential in
  temperature and a too-narrow summer interval is the risky case.

Honest caveats:

- The calibration window must span a **full seasonal cycle** (the default 365
  days). A single-season calibration over- or under-corrects on other seasons.
- No method makes every season hit 0.80 at once. In that run the test year's
  winter was harder than prior winters (year-over-year drift, which conformal
  cannot remove), so under `"mondrian"` winter sat near 0.73. Adding more
  calibration years did not change this, confirming it is drift, not sampling.
  Winter aging is low and benign, so this is the less consequential season to
  under-cover.

Calibration changes only the interval; the `p50` point forecast and the skill
scores are unchanged. A calibrated model carries its adjustment through
`save`/`load`.

## Where to run it

The module is environment-agnostic. Training data is tiny (under ~100 MB even
at the full ~25-year NASA POWER record), so it runs on a laptop CPU; no GPU is
used or needed. For a memory-constrained machine, train on a free Kaggle or
Colab notebook instead, and cache the daily panel (`ForecastConfig.cache_dir`,
or a Kaggle Dataset) so you fetch once and reuse.

## Saving and reusing a trained model

Training and inference are the same call by default (`forecast` fits a fresh
model each time). To train once on long history and reuse the model cheaply,
save it and reload it:

```python
from climagrid.forecasting import ForecastConfig
from climagrid.forecasting.dataset import build_supervised_frame, build_training_panel
from climagrid.forecasting.models import LightGBMForecaster

config = ForecastConfig()
panel = build_training_panel("my_assets.csv", start, end, config)
model = LightGBMForecaster(config).fit(build_supervised_frame(panel, config.targets[0], config), config.targets[0])
model.save("thermal_model.joblib")

# later, anywhere:
model = LightGBMForecaster.load("thermal_model.joblib")
```

Inference does **not** need the full training history. The predictors are
autoregressive lags and trailing rolling windows that reach back at most
`config.min_inference_history_days` (about 30 days with the defaults), so to
forecast forward you only need each asset's most recent ~30 days, not the years
the model was trained on.

The Kaggle training notebook (`examples/kaggle_training.ipynb`) puts this
together end to end: it fetches the full history once (cached), trains on
10-year, 15-year and full-record windows, backtests all three on the same recent
period to pick the most accurate, and saves every model for download.

## Command line

```bash
# Forward forecast
climagrid forecast --assets my_assets.csv --horizon 7 -o forecast.parquet

# Rolling-origin skill-score table instead of a forward forecast
climagrid forecast --assets my_assets.csv --backtest -o skill.csv
```
