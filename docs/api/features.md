# Features

All feature classes follow the same interface: instantiate, then call `.compute(df)` which returns a copy of the DataFrame with new `feat_` columns added.

## ThermalStressIndex

```{eval-rst}
.. autoclass:: climagrid.features.thermal.ThermalStressIndex
   :members:
```

## ConductorSagIndex

```{eval-rst}
.. autoclass:: climagrid.features.conductor_sag.ConductorSagIndex
   :members:
```

## FreezeThawtCycleCounter

```{eval-rst}
.. autoclass:: climagrid.features.freeze_thaw.FreezeThawtCycleCounter
   :members:
```

## IceLoadingRisk

```{eval-rst}
.. autoclass:: climagrid.features.ice_loading.IceLoadingRisk
   :members:
```

## SoilSaturationIndex

```{eval-rst}
.. autoclass:: climagrid.features.soil.SoilSaturationIndex
   :members:
```

## WildfireProximityScore

```{eval-rst}
.. autoclass:: climagrid.features.wildfire.WildfireProximityScore
   :members:
```
