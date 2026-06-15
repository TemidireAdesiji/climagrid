# Validation Notes

This page records, honestly, what climagrid has and has not been shown to do. It exists so that users and reviewers do not have to guess.

## What is verified

climagrid correctly implements published engineering standards (IEEE C57.91, IEEE 738, ASCE 7-22) and applies them to free public weather data to produce interpretable, per-asset environmental stress scores. The outputs are traceable: every score maps to a named standard and a specific public input. The toolkit runs nationwide on any U.S. location and is deterministic and reproducible.

## What is not yet validated

climagrid does not predict equipment failures or outages, and it has not been shown to do so. The stress scores are physically motivated inputs to that question, not an answer to it.

We explored validating the scores against the EAGLE-I public outage dataset (Oak Ridge National Laboratory, county-level customers-without-power). The results did not support a generalizable predictive relationship:

- For Winter Storm Uri (Texas, February 2021), per-customer outage severity correlated with the ice-loading feature (a roughly 3.75x lift from the lowest to highest stress quintile).
- That relationship did not replicate elsewhere. For the June 2021 Pacific Northwest heat dome, the thermal-aging feature showed no positive relationship to outage severity (Washington was strongly negative).
- In a daily outage-occurrence model, adding climagrid features to raw weather produced essentially no predictive lift (within noise across Texas, Washington, Oregon, and California).

The reason is that county-level outages are confounded by population, acute storms, electricity demand, and grid-operator load shedding, none of which is chronic asset stress. County outages are therefore not a clean ground truth for what climagrid measures.

## What validation would require

A genuine test of "does high stress predict failure" needs **asset-level failure records** (which specific transformer or line segment failed, and when). That data lives in individual utilities' outage-management and maintenance systems and is not publicly available. Establishing the link is a deployment-and-partnership effort with a utility, and is explicitly future work.

## How to use this honestly

Use climagrid's scores as **interpretable, standards-based prioritization inputs**: which assets are under the most weather stress, and why. To turn them into failure predictions, combine them with your own historical failure records and validate on your own data.

## Forecasting

The [forecasting module](forecasting.md) extends this stance forward in time: it forecasts a stress feature's future value, not a future failure. The same honesty rules apply.

- The target is a daily stress feature (for example heat-aging stress), which is self-supervised: the label is simply the feature's own value on a future day, computed by climagrid. No failure data is used or implied.
- Every forecast is benchmarked against two naive baselines, persistence (tomorrow equals today) and climatology (the day-of-year average), in a rolling-origin backtest with an embargo gap so a training target window never overlaps a test predictor window. The model is only worth using on horizons and targets where it actually beats those baselines, and the skill-score table reports this honestly, including where it does not.
- Because stress features are smooth and autocorrelated, persistence is a strong baseline at short lead times; meaningful skill is most likely at medium range.
- How much history to train on is decided empirically by `history_ablation`, not assumed. The oldest years reflect a slightly different climate (the global surface record has warmed roughly 0.2 C per decade since the early 1980s), but whether that materially affects a given asset's forecast is measured, not asserted.

A forecast of rising environmental stress is a planning aid for inspection timing. It is still not a failure prediction.

### First backtest results (thermal aging)

A first run on 33 real U.S. substations forecast `feat_thermal_aging_factor` (IEEE C57.91) one to seven days ahead, using NASA POWER hourly history aggregated to daily, a LightGBM quantile model, and a three-fold rolling-origin backtest with an embargo gap.

The model beat both baselines at every horizon:

| Horizon (days) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Skill vs persistence | 0.10 | 0.25 | 0.34 | 0.38 | 0.41 | 0.45 | 0.46 |
| Skill vs climatology | 0.89 | 0.80 | 0.78 | 0.76 | 0.75 | 0.76 | 0.75 |

Skill is positive throughout and grows with lead time, so the model adds the most value exactly where the lead time is useful. Feature-importance analysis shows the edge over persistence comes from day-of-year seasonality (about 22 percent of importance) and the 30-day trailing context, signals that persistence ignores.

How much history helped was decided empirically. A 10, 15, and 25-year ablation came out essentially tied on skill versus persistence (0.34, 0.33, 0.33), with 10 years marginally best. More history added no measurable value, so about 10 years is the recommended default. This is consistent with the mild non-stationarity noted above, now measured rather than assumed.

The raw 80 percent (p10 to p90) prediction intervals covered 0.75 to 0.78 of outcomes against the 0.80 target. Conformal calibration (`calibrate_intervals=True`, the season-conditional Mondrian method by default) closes most of this: it brings overall coverage to about 0.78 and the high-stress summer interval to target. The test year's winter stayed near 0.73 because that winter was more variable than prior years, year-over-year drift that conformal calibration cannot remove.

Caveats: these results are for thermal aging only. The cumulative or rolling features (heat hours, freeze-thaw, soil saturation) are expected to show little skill over persistence and have not yet been evaluated. Errors were largest on hot-climate assets (Texas), where thermal stress is highest and most variable, and smallest on mild Pacific Northwest assets. This remains a forecast of environmental stress for inspection lead time, not a failure prediction.
