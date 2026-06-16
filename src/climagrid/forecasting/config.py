"""
Configuration for the climagrid forecasting module.

ForecastConfig is the single source of truth for every tunable knob in the
forecast pipeline: which stress features to forecast, how far ahead, the
predictor windows, the model family, and the resource-bounding defaults.

The defaults are deliberately conservative so the pipeline runs on a
memory-constrained laptop; see docs/forecasting.md for the rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Stress-feature target -> the climagrid feature name that produces it.
# Used to fetch only the features a forecast actually needs.
TARGET_TO_FEATURE: dict[str, str] = {
    "feat_thermal_aging_factor": "thermal",
    "feat_heat_hours_above_35c": "thermal",
    "feat_freeze_thaw_cycles": "freeze_thaw",
    "feat_ice_loading_risk": "ice_loading",
    "feat_soil_saturation_index": "soil",
    "feat_wildfire_proximity": "wildfire",
    "feat_conductor_sag_index": "conductor_sag",
}

DailyAgg = Literal["max", "mean", "min"]
ModelName = Literal["lightgbm", "lstm", "persistence", "climatology"]
CalibrationMethod = Literal["normalized", "constant", "mondrian"]


class LSTMParams(BaseModel):
    """
    Hyperparameters for the optional LSTM forecaster (the ``[dl]`` extra).

    These are read only by ``LSTMForecaster`` and ignored by the LightGBM
    model, so a single ``ForecastConfig`` can drive both for an apples-to-apples
    benchmark. The LSTM consumes the same supervised frame as LightGBM: its
    encoder sequence is the contiguous lag window ``[lag_max .. lag_1, y_t]``,
    so both models see exactly the same predictor information. For a clean
    sequence, train with contiguous lags (e.g. ``lags=list(range(1, 29))``).

    Parameters
    ----------
    hidden_size:
        Width of the LSTM hidden state.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout applied between LSTM layers and before the output head.
    epochs:
        Maximum training epochs (early stopping may end training sooner).
    batch_size:
        Minibatch size.
    learning_rate:
        Adam learning rate.
    weight_decay:
        Adam L2 regularization.
    patience:
        Early-stopping patience in epochs on the validation pinball loss.
    val_fraction:
        Fraction of the most recent training origins held out for early
        stopping (kept chronological, never shuffled, to avoid leakage).
    seed:
        Base random seed. Multi-seed runs vary this to report mean and spread.
    device:
        ``"auto"`` uses CUDA when available, else CPU. Override for tests.
    """

    model_config = ConfigDict(frozen=True)

    hidden_size: int = Field(default=64, ge=1)
    num_layers: int = Field(default=1, ge=1, le=4)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    epochs: int = Field(default=60, ge=1)
    batch_size: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    patience: int = Field(default=8, ge=1)
    val_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = Field(default=42)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class ForecastConfig(BaseModel):
    """
    Parameters controlling the stress-feature forecast pipeline.

    Parameters
    ----------
    targets:
        Stress-feature columns to forecast. Default is the single headline
        target ``feat_thermal_aging_factor`` (a per-row Arrhenius function of
        temperature, where an ML model has the most to add over naive
        baselines). Add more targets at the cost of more trained models.
    horizon_days:
        Maximum forecast lead time in days. One model is trained per horizon
        ``1..horizon_days`` (direct multi-horizon strategy).
    daily_agg:
        How hourly stress features are reduced to one value per day.
        Default ``"max"`` matches how outputs.report.rank_assets ranks assets
        by peak stress.
    history_years:
        Default length of training history when explicit dates are not given.
        15 years balances seasonal-cycle coverage against climate drift in the
        oldest years; the right value is confirmed by backtest.history_ablation.
    lags:
        Target autoregressive lags (in days) used as predictors.
    rolling_windows:
        Trailing-window sizes (in days) for rolling mean/std predictors.
    quantiles:
        Quantiles to predict for prediction intervals. Must include 0.5.
    model:
        Forecaster family. ``"lightgbm"`` is the trained model; the other two
        are the baselines the model is scored against.
    per_asset:
        If True, fit one model per asset; otherwise a single global pooled
        model that generalizes to unseen asset locations (default).
    embargo_days:
        Gap in days between train and test in the rolling-origin backtest, so
        a training target window never overlaps a test predictor window.
        Defaults to ``horizon_days`` when None.
    climatology_window_years:
        If set, the climatology baseline uses only the most recent N years of
        history (a hedge against climate drift). None uses all training years.
    calibrate_intervals:
        If True, conformally calibrate the prediction intervals: hold out the
        most recent ``calibration_days`` as a calibration set, fit on the rest,
        and adjust the interval so its coverage matches the nominal level.
    calibration_method:
        ``"mondrian"`` (default): a separate width per meteorological season,
        which keeps the high-stress summer interval on target. ``"normalized"``:
        locally adaptive width (even overall, but summer under-covered).
        ``"constant"``: a single additive width per horizon.
    calibration_days:
        Size of the held-out calibration window. Should span a full seasonal
        cycle (default 365) so the calibration is not biased to one season.
    sources:
        climagrid data sources to fetch. Default ``["nasa_power"]`` (keyless,
        hourly back to 2001).
    n_jobs, n_estimators, num_leaves, learning_rate, random_state:
        LightGBM training parameters; defaults are conservative for a
        memory-constrained machine.
    cache_dir:
        Directory for the cached daily panel parquet. None disables caching.
    """

    model_config = ConfigDict(frozen=True)

    targets: list[str] = Field(default_factory=lambda: ["feat_thermal_aging_factor"])
    horizon_days: int = Field(default=7, ge=1, le=60)
    daily_agg: DailyAgg = "max"
    history_years: int = Field(default=15, ge=1, le=25)
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 7, 14, 30])
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 30])
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    model: ModelName = "lightgbm"
    per_asset: bool = False
    embargo_days: int | None = Field(default=None, ge=0)
    climatology_window_years: int | None = Field(default=None, ge=1)
    calibrate_intervals: bool = False
    calibration_method: CalibrationMethod = "mondrian"
    calibration_days: int = Field(default=365, ge=30)
    sources: list[str] = Field(default_factory=lambda: ["nasa_power"])

    n_jobs: int = Field(default=2, ge=1)
    n_estimators: int = Field(default=300, ge=1)
    num_leaves: int = Field(default=31, ge=2)
    learning_rate: float = Field(default=0.05, gt=0.0)
    random_state: int = 42

    cache_dir: Path | None = None

    lstm: LSTMParams = Field(default_factory=LSTMParams)

    @field_validator("targets")
    @classmethod
    def _known_targets(cls, value: list[str]) -> list[str]:
        unknown = [t for t in value if t not in TARGET_TO_FEATURE]
        if unknown:
            raise ValueError(
                f"Unknown forecast target(s): {unknown}. "
                f"Valid: {sorted(TARGET_TO_FEATURE)}"
            )
        if not value:
            raise ValueError("At least one forecast target is required.")
        return value

    @field_validator("quantiles")
    @classmethod
    def _quantiles_include_median(cls, value: list[float]) -> list[float]:
        if any(not 0.0 < q < 1.0 for q in value):
            raise ValueError("Quantiles must lie strictly between 0 and 1.")
        if 0.5 not in value:
            raise ValueError("Quantiles must include 0.5 (the point forecast).")
        return sorted(value)

    @property
    def effective_embargo_days(self) -> int:
        """Embargo gap, defaulting to the max horizon when unset."""
        return self.horizon_days if self.embargo_days is None else self.embargo_days

    @property
    def min_inference_history_days(self) -> int:
        """Days of recent history needed to build predictors at inference time.

        Equals the largest lag or rolling window. Forecasting forward from a
        saved model only needs this many recent days per asset, not the full
        training history, because the predictors are autoregressive lags and
        trailing rolling statistics that reach back at most this far.
        """
        return max(max(self.lags), max(self.rolling_windows))

    def required_features(self) -> list[str]:
        """climagrid feature names needed to compute the configured targets."""
        return sorted({TARGET_TO_FEATURE[t] for t in self.targets})
