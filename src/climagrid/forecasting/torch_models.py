"""
LSTM quantile forecaster (the optional ``[dl]`` extra).

A compact recurrent sequence-to-one model benchmarked against
``LightGBMForecaster`` in the same backtest harness. It implements the same
``fit`` / ``predict`` / ``calibrate`` / ``save`` / ``load`` interface, consumes the
same supervised frame, predicts the same quantiles, and reuses the same
conformal calibration (``climagrid.forecasting.conformal``), so the comparison
between trees and a neural net is apples-to-apples.

Architecture: an LSTM encodes the contiguous target history
``[lag_max .. lag_1, y_t]``; its final hidden state is concatenated with static
covariates (day-of-year harmonics and location) and passed through a linear head
that emits every ``(horizon, quantile)`` directly (the direct multi-horizon
strategy, matching the LightGBM model). The model is trained with pinball loss
summed over horizons and quantiles, with chronological early stopping.

PyTorch is imported eagerly here; this module is only imported when the ``[dl]``
extra is installed. Importing ``climagrid.forecasting`` does not import it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climagrid.forecasting import conformal as conformal_mod
from climagrid.forecasting.config import ForecastConfig
from climagrid.forecasting.models import quantile_column_names
from climagrid.forecasting.windows import (
    STATIC_COLUMNS,
    SequenceScaler,
    build_arrays,
    fit_scaler,
)

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "PyTorch is required for the LSTM forecaster. "
        'Install it with: pip install "climagrid[dl]"'
    ) from exc

logger = logging.getLogger(__name__)


class _LSTMNet(nn.Module):
    """LSTM encoder + static covariates -> direct multi-horizon quantile head."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        static_dim: int,
        n_outputs: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + static_dim, n_outputs)

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(sequence)
        last = h_n[-1]  # final layer's hidden state, shape (batch, hidden)
        combined = torch.cat([last, static], dim=1)
        return self.head(self.dropout(combined))  # type: ignore[no-any-return]


def _pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    """Masked pinball loss averaged over valid (row, horizon, quantile) entries.

    ``pred`` is ``(batch, horizon, quantile)``; ``target`` and ``mask`` are
    ``(batch, horizon)``. Masked-out (NaN-target) entries contribute nothing.
    """
    target_e = target.unsqueeze(-1)
    mask_e = mask.unsqueeze(-1)
    diff = target_e - pred
    loss = torch.maximum(quantiles * diff, (quantiles - 1.0) * diff) * mask_e
    denom = mask_e.sum() * pred.shape[-1]
    return loss.sum() / denom.clamp(min=1.0)


class LSTMForecaster:
    """Direct multi-horizon quantile forecaster backed by an LSTM."""

    def __init__(self, config: ForecastConfig):
        self._config = config
        self._params = config.lstm
        self._quantiles = config.quantiles
        self._horizons = config.horizon_days
        self._n_outputs = self._horizons * len(self._quantiles)
        self._target: str | None = None
        self._scaler: SequenceScaler | None = None
        self._net: _LSTMNet | None = None
        self._device = self._resolve_device(self._params.device)
        # Conformal state, identical contract to LightGBMForecaster.
        self._conformal: dict[tuple[int, int], float] = {}
        self._conformal_scale: dict[tuple[int, int], float] = {}
        self._conformal_method: str = ""

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _new_net(self) -> _LSTMNet:
        return _LSTMNet(
            hidden_size=self._params.hidden_size,
            num_layers=self._params.num_layers,
            dropout=self._params.dropout,
            static_dim=len(STATIC_COLUMNS),
            n_outputs=self._n_outputs,
        ).to(self._device)

    def _target_columns(self) -> list[str]:
        return [f"y_h{h}" for h in range(1, self._horizons + 1)]

    def fit(self, frame: pd.DataFrame, target: str) -> LSTMForecaster:
        """Train the LSTM with pinball loss and chronological early stopping."""
        torch.manual_seed(self._params.seed)
        np.random.seed(self._params.seed)
        self._target = target
        scaler = fit_scaler(frame, self._config)
        self._scaler = scaler

        df = frame.sort_values(["asset_id", "date"]).reset_index(drop=True)
        sequence, static = build_arrays(df, self._config, scaler)
        y_raw = df[self._target_columns()].to_numpy(dtype=float)
        mask = np.isfinite(y_raw)
        y_scaled = np.where(mask, (y_raw - scaler.seq_mean) / scaler.seq_std, 0.0)

        train_idx, val_idx = self._chronological_split(df)
        net = self._new_net()
        quantiles_t = torch.tensor(
            self._quantiles, dtype=torch.float32, device=self._device
        )

        def _to_loader(idx: np.ndarray, shuffle: bool) -> Any:
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(sequence[idx], dtype=torch.float32),
                torch.tensor(static[idx], dtype=torch.float32),
                torch.tensor(y_scaled[idx], dtype=torch.float32),
                torch.tensor(mask[idx], dtype=torch.float32),
            )
            generator = torch.Generator().manual_seed(self._params.seed)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self._params.batch_size,
                shuffle=shuffle,
                generator=generator if shuffle else None,
            )

        train_loader = _to_loader(train_idx, shuffle=True)
        val_loader = _to_loader(val_idx, shuffle=False) if len(val_idx) else None
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=self._params.learning_rate,
            weight_decay=self._params.weight_decay,
        )

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        epochs_no_improve = 0
        for epoch in range(self._params.epochs):
            net.train()
            for seq_b, stat_b, y_b, m_b in train_loader:
                seq_b = seq_b.to(self._device)
                stat_b = stat_b.to(self._device)
                y_b = y_b.to(self._device)
                m_b = m_b.to(self._device)
                optimizer.zero_grad()
                out = net(seq_b, stat_b).view(-1, self._horizons, len(self._quantiles))
                loss = _pinball_loss(out, y_b, m_b, quantiles_t)
                loss.backward()
                optimizer.step()

            monitor = val_loader or train_loader
            val_loss = self._epoch_loss(net, monitor, quantiles_t)
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self._params.patience:
                    logger.info("Early stopping at epoch %d (val %.5f).", epoch, best_loss)
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._net = net
        return self

    def _chronological_split(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Hold out the most recent origin dates for early stopping (no leakage)."""
        unique_dates = np.sort(df["date"].unique())
        if len(unique_dates) < 2:
            return np.arange(len(df)), np.array([], dtype=int)
        n_val = max(1, int(len(unique_dates) * self._params.val_fraction))
        val_dates = set(unique_dates[-n_val:])
        is_val = df["date"].isin(val_dates).to_numpy()
        return np.where(~is_val)[0], np.where(is_val)[0]

    def _epoch_loss(self, net: _LSTMNet, loader: Any, quantiles_t: torch.Tensor) -> float:
        net.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for seq_b, stat_b, y_b, m_b in loader:
                seq_b = seq_b.to(self._device)
                stat_b = stat_b.to(self._device)
                y_b = y_b.to(self._device)
                m_b = m_b.to(self._device)
                out = net(seq_b, stat_b).view(-1, self._horizons, len(self._quantiles))
                total += float(_pinball_loss(out, y_b, m_b, quantiles_t)) * len(seq_b)
                count += len(seq_b)
        return total / max(count, 1)

    def _raw_predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Unscaled quantile predictions, shape ``(n_rows, horizon, quantile)``."""
        if self._net is None or self._scaler is None:
            raise RuntimeError("LSTMForecaster.predict called before fit.")
        sequence, static = build_arrays(frame, self._config, self._scaler)
        self._net.eval()
        with torch.no_grad():
            out = self._net(
                torch.tensor(sequence, dtype=torch.float32, device=self._device),
                torch.tensor(static, dtype=torch.float32, device=self._device),
            )
        preds = out.cpu().numpy().reshape(len(frame), self._horizons, len(self._quantiles))
        return preds * self._scaler.seq_std + self._scaler.seq_mean  # type: ignore[no-any-return]

    def calibrate(
        self,
        frame: pd.DataFrame,
        target: str | None = None,
        *,
        method: str = "normalized",
    ) -> LSTMForecaster:
        """Conformalize the outer interval, reusing the shared CQR logic.

        Identical contract and math to ``LightGBMForecaster.calibrate``; see
        ``climagrid.forecasting.conformal``.
        """
        if self._net is None:
            raise RuntimeError("LSTMForecaster.calibrate called before fit.")
        level = self._quantiles[-1] - self._quantiles[0]
        raw = self._raw_predict(frame)

        per_horizon: dict[int, conformal_mod.HorizonCalib] = {}
        for h in range(1, self._horizons + 1):
            y_all = frame[f"y_h{h}"]
            keep = y_all.notna().to_numpy()
            if keep.sum() == 0:
                continue
            lo = raw[keep, h - 1, 0]
            hi = raw[keep, h - 1, -1]
            y_h = y_all[keep].to_numpy(dtype=float)
            forecast_dates = pd.to_datetime(frame.loc[keep, "date"]) + pd.to_timedelta(
                h, unit="D"
            )
            per_horizon[h] = (lo, hi, y_h, forecast_dates)

        self._conformal, self._conformal_scale = conformal_mod.fit_conformal(
            method, level, per_horizon
        )
        self._conformal_method = method
        return self

    def predict(self, frame: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
        """Long-form forecasts per (row, horizon); see LightGBMForecaster.predict."""
        target_name = target or self._target or ""
        q_cols = quantile_column_names(self._config)
        raw = self._raw_predict(frame)

        records: list[pd.DataFrame] = []
        for h in range(1, self._horizons + 1):
            preds = np.sort(raw[:, h - 1, :], axis=1)
            if self._conformal_method:
                forecast_dates = pd.to_datetime(frame["date"]) + pd.to_timedelta(
                    h, unit="D"
                )
                delta = conformal_mod.conformal_delta(
                    self._conformal_method,
                    self._conformal,
                    self._conformal_scale,
                    h,
                    preds,
                    forecast_dates,
                )
                preds[:, 0] -= delta
                preds[:, -1] += delta
                preds = np.sort(preds, axis=1)
            block = pd.DataFrame(
                {
                    "asset_id": frame["asset_id"].to_numpy(),
                    "origin_date": frame["date"].to_numpy(),
                    "horizon_day": h,
                    "target": target_name,
                }
            )
            block["forecast_date"] = block["origin_date"] + pd.to_timedelta(h, unit="D")
            for i, col in enumerate(q_cols):
                block[col] = preds[:, i]
            records.append(block)

        if not records:
            return pd.DataFrame()
        ordered_cols = [
            "asset_id",
            "origin_date",
            "forecast_date",
            "horizon_day",
            "target",
            *q_cols,
        ]
        return pd.concat(records, ignore_index=True)[ordered_cols]  # type: ignore[no-any-return]

    def save(self, path: str | Path) -> Path:
        """Persist weights, scaler, config and conformal state (torch.save)."""
        if self._net is None or self._scaler is None:
            raise RuntimeError("Cannot save an LSTMForecaster before fit.")
        path = Path(path)
        torch.save(
            {
                "config": self._config,
                "target": self._target,
                "scaler": self._scaler.model_dump(),
                "state_dict": self._net.state_dict(),
                "conformal": self._conformal,
                "conformal_scale": self._conformal_scale,
                "conformal_method": self._conformal_method,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> LSTMForecaster:
        """Reload a forecaster saved by :meth:`save`."""
        state = torch.load(Path(path), weights_only=False)
        forecaster = cls(state["config"])
        forecaster._target = state["target"]
        forecaster._scaler = SequenceScaler(**state["scaler"])
        net = forecaster._new_net()
        net.load_state_dict(state["state_dict"])
        net.eval()
        forecaster._net = net
        forecaster._conformal = state.get("conformal", {})
        forecaster._conformal_scale = state.get("conformal_scale", {})
        forecaster._conformal_method = state.get("conformal_method", "")
        return forecaster
