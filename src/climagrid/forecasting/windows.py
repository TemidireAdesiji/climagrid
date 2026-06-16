"""
Sequence construction for the LSTM forecaster.

Turns the same supervised frame that LightGBM consumes into the tensors a
recurrent model needs, so the two models are trained on identical information
and the benchmark is apples-to-apples.

For each supervised row the encoder sequence is the contiguous recent history of
the target, ordered oldest to newest: ``[lag_max, .., lag_1, y_t]``. Static
covariates (day-of-year harmonics and location) are passed alongside the
sequence rather than per timestep. Train with contiguous lags
(``lags=list(range(1, L + 1))``) so the sequence is a true daily window with no
gaps; sparse lags still work but the sequence is then unevenly spaced.

Scaling is fit on the training frame only (mean/std of the target channel and of
each static column) and reused at predict time, so no test statistics leak into
training. The target columns ``y_h*`` share the sequence channel's units and are
scaled by the same constants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from climagrid.forecasting.config import ForecastConfig

# Static (non-sequential) predictor columns, known at the forecast origin.
STATIC_COLUMNS = ["doy_sin", "doy_cos", "lat", "lon"]


def sequence_columns(config: ForecastConfig) -> list[str]:
    """Encoder-sequence columns, ordered oldest to newest (``y_t`` last)."""
    return [f"lag_{lag}" for lag in sorted(config.lags, reverse=True)] + ["y_t"]


class SequenceScaler(BaseModel):
    """Standardization constants fit on the training frame (target + statics)."""

    model_config = ConfigDict(frozen=True)

    seq_mean: float
    seq_std: float
    static_mean: list[float]
    static_std: list[float]


def _safe_std(value: float) -> float:
    """Avoid division by zero for a constant column."""
    return value if value > 1e-8 else 1.0


def fit_scaler(frame: pd.DataFrame, config: ForecastConfig) -> SequenceScaler:
    """Fit standardization constants on a training frame."""
    seq_cols = sequence_columns(config)
    seq_values = frame[seq_cols].to_numpy(dtype=float)
    seq_mean = float(np.nanmean(seq_values))
    seq_std = _safe_std(float(np.nanstd(seq_values)))

    static = frame[STATIC_COLUMNS].to_numpy(dtype=float)
    static_mean = np.nanmean(static, axis=0)
    static_std = np.array([_safe_std(float(s)) for s in np.nanstd(static, axis=0)])
    return SequenceScaler(
        seq_mean=seq_mean,
        seq_std=seq_std,
        static_mean=[float(m) for m in static_mean],
        static_std=[float(s) for s in static_std],
    )


def _fill_time_gaps(seq: np.ndarray) -> np.ndarray:
    """Fill NaNs along the time axis of ``(n, L)`` by nearest valid value.

    Leading NaNs (early in an asset's history, before enough lags exist) are
    back-filled from the first valid step; any remaining NaNs are forward-filled;
    a fully missing row falls back to zero (the post-scaling mean).
    """
    filled = pd.DataFrame(seq).bfill(axis=1).ffill(axis=1).fillna(0.0)
    return filled.to_numpy(dtype=float)  # type: ignore[no-any-return]


def build_arrays(
    frame: pd.DataFrame, config: ForecastConfig, scaler: SequenceScaler
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the scaled encoder sequence and static arrays for a frame.

    Returns
    -------
    (sequence, static):
        ``sequence`` has shape ``(n_rows, seq_len, 1)`` and ``static`` shape
        ``(n_rows, len(STATIC_COLUMNS))``, both standardized with ``scaler``.
    """
    seq_cols = sequence_columns(config)
    seq = _fill_time_gaps(frame[seq_cols].to_numpy(dtype=float))
    seq = (seq - scaler.seq_mean) / scaler.seq_std
    sequence = seq[:, :, np.newaxis]

    static = frame[STATIC_COLUMNS].to_numpy(dtype=float)
    mean = np.asarray(scaler.static_mean, dtype=float)
    std = np.asarray(scaler.static_std, dtype=float)
    static = np.where(np.isnan(static), mean, static)
    static = (static - mean) / std
    return sequence, static
