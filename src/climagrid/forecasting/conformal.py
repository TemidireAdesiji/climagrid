"""
Shared conformal calibration of prediction intervals (Romano et al. 2019, CQR).

The math is model-agnostic: it consumes a model's own lower/upper quantile
predictions on a held-out calibration set and returns the per-horizon widening
needed to attain nominal coverage out of sample. Both ``LightGBMForecaster`` and
``LSTMForecaster`` call these same functions, so the two are calibrated by
identical logic and the interval comparison between them is apples-to-apples.

Three methods, trading marginal versus per-season coverage:

- ``"constant"``: a single additive width ``Q`` per horizon.
- ``"normalized"``: scales ``Q`` by the model's own interval width so the
  widening adapts to local uncertainty.
- ``"mondrian"``: a separate additive ``Q`` per meteorological season (keyed on
  the forecast date), with a pooled per-horizon fallback for sparse seasons.

State is held by the caller as two dicts keyed by ``(horizon, season_bin)``:
bin 0 for ``"constant"``/``"normalized"``; bins 0-3 (DJF/MAM/JJA/SON) plus a -1
pooled fallback for ``"mondrian"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Minimum calibration points before a per-season width is trusted over the
# pooled per-horizon width.
MIN_SEASON_BIN = 50

# Per-horizon calibration inputs: (lo_pred, hi_pred, y_true, forecast_dates).
HorizonCalib = tuple[np.ndarray, np.ndarray, np.ndarray, Any]


def season_index(dates: Any) -> np.ndarray:
    """Meteorological season per date: 0=DJF, 1=MAM, 2=JJA, 3=SON."""
    months = pd.DatetimeIndex(pd.to_datetime(dates)).month.to_numpy()
    return ((months % 12) // 3).astype(int)


def _emp_quantile(scores: np.ndarray, level: float) -> float:
    """The ``ceil((n + 1) * level) / n`` empirical quantile of conformity scores."""
    n = len(scores)
    if n == 0:
        return 0.0
    rank = min(max(int(np.ceil((n + 1) * level)), 1), n)
    return float(np.sort(scores)[rank - 1])


def fit_conformal(
    method: str,
    level: float,
    per_horizon: dict[int, HorizonCalib],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """
    Compute conformal widths from per-horizon calibration predictions.

    Parameters
    ----------
    method:
        ``"constant"``, ``"normalized"`` or ``"mondrian"``.
    level:
        Nominal interval coverage (``q_hi - q_lo``, e.g. 0.80 for p10-p90).
    per_horizon:
        Maps horizon ``h`` to ``(lo, hi, y_true, forecast_dates)``, the model's
        lower/upper quantile predictions, the realized targets, and the forecast
        dates (``origin + h``) used for the Mondrian seasonal split.

    Returns
    -------
    (conformal, conformal_scale):
        The width dict and, for ``"normalized"`` only, the per-horizon scale
        constant; both keyed by ``(horizon, season_bin)``.
    """
    if method not in {"constant", "normalized", "mondrian"}:
        raise ValueError(f"Unknown calibration method: {method!r}")

    conformal: dict[tuple[int, int], float] = {}
    conformal_scale: dict[tuple[int, int], float] = {}

    for h, (lo, hi, y_true, forecast_dates) in per_horizon.items():
        raw = np.maximum(lo - y_true, y_true - hi)
        if method == "normalized":
            width = hi - lo
            c = float(np.median(width))
            conformal_scale[(h, 0)] = c
            conformal[(h, 0)] = _emp_quantile(raw / (width + c), level)
        elif method == "constant":
            conformal[(h, 0)] = _emp_quantile(raw, level)
        else:  # mondrian: additive Q per season of the forecast (target) date
            seasons = season_index(forecast_dates)
            pooled = _emp_quantile(raw, level)
            conformal[(h, -1)] = pooled  # fallback for sparse seasons
            for season in np.unique(seasons):
                in_season = raw[seasons == season]
                conformal[(h, int(season))] = (
                    _emp_quantile(in_season, level)
                    if len(in_season) >= MIN_SEASON_BIN
                    else pooled
                )

    return conformal, conformal_scale


def conformal_delta(
    method: str,
    conformal: dict[tuple[int, int], float],
    conformal_scale: dict[tuple[int, int], float],
    h: int,
    preds: np.ndarray,
    forecast_dates: Any,
) -> float | np.ndarray:
    """
    Per-row interval widening for horizon ``h`` under the active method.

    ``preds`` is the row-sorted quantile matrix (column 0 is the lowest
    quantile, column -1 the highest). ``forecast_dates`` are the per-row forecast
    dates (``origin + h``), used only by ``"mondrian"``.
    """
    if method == "normalized":
        q = conformal.get((h, 0))
        if q is None:
            return 0.0
        interval = preds[:, -1] - preds[:, 0]
        return q * (interval + conformal_scale.get((h, 0), 0.0))  # type: ignore[no-any-return]
    if method == "mondrian":
        seasons = season_index(forecast_dates)
        fallback = conformal.get((h, -1), 0.0)
        return np.array(
            [conformal.get((h, int(s)), fallback) for s in seasons],
            dtype=float,
        )
    # constant
    return conformal.get((h, 0), 0.0)
