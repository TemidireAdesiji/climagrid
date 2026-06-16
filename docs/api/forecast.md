# Forecasting

Medium-range probabilistic forecasting of asset stress features. Requires the `[ml]` extra (`pip install "climagrid[ml]"`). See the {doc}`forecasting guide </forecasting>` for an overview.

## climagrid.forecast

```{eval-rst}
.. autofunction:: climagrid.forecasting.api.forecast
```

## ForecastConfig

```{eval-rst}
.. autoclass:: climagrid.forecasting.config.ForecastConfig
   :members:
```

## Dataset construction

```{eval-rst}
.. autofunction:: climagrid.forecasting.dataset.build_training_panel

.. autofunction:: climagrid.forecasting.dataset.build_supervised_frame
```

## Model and baselines

```{eval-rst}
.. autoclass:: climagrid.forecasting.models.LightGBMForecaster
   :members:

.. autoclass:: climagrid.forecasting.baselines.PersistenceForecaster
   :members:

.. autoclass:: climagrid.forecasting.baselines.ClimatologyForecaster
   :members:
```

## Backtesting

```{eval-rst}
.. autofunction:: climagrid.forecasting.backtest.evaluate

.. autofunction:: climagrid.forecasting.backtest.rolling_origin_splits

.. autofunction:: climagrid.forecasting.backtest.history_ablation
```
