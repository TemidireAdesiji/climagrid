# Getting Started

## Installation

climagrid requires Python 3.10 or later.

```bash
pip install climagrid
```

For NOAA HRRR support (requires GRIB2 libraries):

```bash
pip install "climagrid[noaa-nwp]"
```

For all optional dependencies:

```bash
pip install "climagrid[full]"
```

---

## Quickstart: one call to `climagrid.run()`

The simplest path: give it an asset CSV and a time range, get back a feature matrix.

```python
import climagrid
from datetime import datetime, timezone

df = climagrid.run(
    "assets.csv",                                   # your asset file
    start_dt=datetime(2024, 7, 1, tzinfo=timezone.utc),
    end_dt=datetime(2024, 7, 8, tzinfo=timezone.utc),
    sources=["nasa_power"],                         # data sources to fetch
    features="all",                                 # compute all stress features
)

print(df.shape)        # (n_assets × n_hours, n_columns)
print(df.columns.tolist())
```

---

## Asset file format

Your asset CSV must have at minimum three columns:

| Column | Type | Description |
|--------|------|-------------|
| `asset_id` | string | Unique identifier (e.g. `"TX-001"`) |
| `lat` | float | WGS-84 latitude |
| `lon` | float | WGS-84 longitude |

Optional columns (`asset_type`, `voltage_kv`, `install_year`) are passed through to the output.

```text
asset_id,lat,lon,asset_type
TX-001,31.5494,-97.1467,transformer
TX-002,31.7621,-97.0542,distribution_line
CO-001,39.7392,-104.9903,substation
```

A sample file of 33 real electric substations (OpenStreetMap, across 7 U.S. states) is included at `examples/data/sample_assets.csv`.

---

## Choosing data sources

Pass a list of source names to `sources=`:

| Source name | Data | API key? | Coverage |
|-------------|------|----------|----------|
| `"nasa_power"` | MERRA-2 hourly surface met | No | Global |
| `"noaa_hrrr"` | HRRR 3 km NWP | No | CONUS only |
| `"noaa_ncei"` | Surface station observations | Yes (free) | Global |
| `"usda_nrcs"` | SCAN/SNOTEL soil sensors | No | CONUS |
| `"usfs_wfigs"` | Active wildfire perimeters | No | CONUS |

For most use cases, `sources=["nasa_power"]` is sufficient. Add `"usfs_wfigs"` if your assets are in wildfire-prone regions.

---

## Choosing features

Pass a list to `features=` or use `"all"`:

| Feature name | Output column | Relevant for |
|---|---|---|
| `"thermal"` | `feat_thermal_aging_factor`, `feat_heat_hours_above_35c` | Transformers |
| `"conductor_sag"` | `feat_conductor_sag_index` | Overhead lines |
| `"freeze_thaw"` | `feat_freeze_thaw_cycles` | Underground cables, pole foundations |
| `"ice_loading"` | `feat_ice_loading_risk` | Overhead lines, towers |
| `"soil"` | `feat_soil_saturation_index` | Underground infrastructure |
| `"wildfire"` | `feat_wildfire_proximity` | All above-ground assets |

---

## Exporting results

```python
from climagrid.outputs import to_csv, to_parquet

to_csv(df, "output/features.csv")
to_parquet(df, "output/features.parquet")  # recommended for >30 days
```

---

## Using individual adapters

For more control, use adapters directly:

```python
from datetime import datetime, timezone
from climagrid.sources.nasa_power import NasaPowerAdapter
from climagrid.sources.base import BoundingBox

adapter = NasaPowerAdapter()
bbox = BoundingBox(min_lat=31.3, max_lat=31.9, min_lon=-97.4, max_lon=-96.9)
start = datetime(2024, 7, 1, tzinfo=timezone.utc)
end   = datetime(2024, 7, 8, tzinfo=timezone.utc)

env_df = adapter.fetch(bbox, start, end)
```

Then join to assets manually:

```python
from climagrid.assets.registry import AssetRegistry
from climagrid.assets.joiner import AssetEnvironmentJoiner

registry = AssetRegistry("assets.csv")
joiner = AssetEnvironmentJoiner()
asset_env_df = joiner.join(registry, env_df)
```

And compute features:

```python
from climagrid.features.thermal import ThermalStressIndex

tsi = ThermalStressIndex()
result = tsi.compute(asset_env_df)
```
