<p align="center">
  <img src="https://raw.githubusercontent.com/TemidireAdesiji/climagrid/main/docs/assets/banner.png" alt="climagrid" />
</p>

<h1 align="center">climagrid</h1>

<p align="center">
  <strong>Turn free public weather data into standards-based stress scores for your grid equipment.</strong><br>
  climagrid is an open-source toolkit built on NOAA, NASA, USDA, and USFS data, the kind of weather-risk analysis that used to require expensive software.
</p>

<p align="center">
  Built for the more than 900 rural cooperatives serving 42 million Americans across 56 percent of U.S. landmass:<br>
  small utilities with no data-science team.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0" /></a>
  <a href="https://pypi.org/project/climagrid/"><img src="https://img.shields.io/pypi/v/climagrid?logo=pypi&logoColor=white&cacheSeconds=1" alt="PyPI" /></a>
  <a href="https://github.com/TemidireAdesiji/climagrid/actions/workflows/ci.yml"><img src="https://github.com/TemidireAdesiji/climagrid/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://climagrid.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/climagrid/badge/?version=latest" alt="Docs" /></a>
  <img src="https://img.shields.io/badge/python-3.10+-3776ab?logo=python&logoColor=white" alt="Python 3.10+" />
  <a href="https://doi.org/10.5281/zenodo.20256535"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20256535-blue.svg" alt="DOI" /></a>
  <a href="https://codecov.io/github/TemidireAdesiji/climagrid"><img src="https://codecov.io/github/TemidireAdesiji/climagrid/graph/badge.svg?token=ACX1C8SW1Q" alt="Coverage" /></a>
</p>

---

## Why this exists

Power outages cost the U.S. economy about **$67 billion every year** on average, according to Oak Ridge National Laboratory. Most of those outages trace back to weather, vegetation, and environmental stress on aging grid equipment: transformers, power lines, and circuit breakers.

Large utilities can afford expensive software to monitor this risk. The **roughly 900 rural electric cooperatives and 2,000 municipal utilities that serve 42 million Americans across 56% of U.S. landmass** mostly cannot. They have small engineering teams, no data scientists, and no budget for six-figure software contracts.

climagrid is built for them. It is free, open source (Apache 2.0), and runs on a single laptop. It takes weather and environmental data that the U.S. government already publishes for free, from NOAA, NASA, the USDA, and the U.S. Forest Service, and turns it into the kind of information a maintenance engineer can actually use: *which of my transformers are under the most heat stress this month? Which power-line spans are closest to active wildfires? Which areas should I prioritize for inspection?*

The U.S. Department of Energy reports that predictive maintenance can cut equipment costs by 25 to 30 percent and return roughly $10 for every $1 invested. climagrid does not deliver those savings on its own; it provides the standards-based environmental stress data that feeds that kind of maintenance planning, for utilities that could not otherwise produce it.

---

## What it does

climagrid pulls from five free federal data sources, joins the data to your asset locations, and computes grid stress features ready to drop into any model or spreadsheet. One call in, one DataFrame out.

![climagrid pipeline](https://raw.githubusercontent.com/TemidireAdesiji/climagrid/main/docs/assets/pipeline.png)

---

## Who this is for

- **Rural electric cooperatives** and **municipal utilities** that want to start using weather and environmental data in their maintenance planning but don't have a data-science team to build the pipeline.
- **Utility engineers** who already run predictive-maintenance or anomaly-detection models and want to add environmental stress features to them as additional inputs.
- **Researchers and journalists** studying grid resilience and rural energy equity.

If you serve fewer than 100,000 meters and your weather "data integration" today is "we check the National Weather Service app before a storm," this toolkit is built for you.

---

## Quick start

```bash
pip install climagrid
```

```python
import climagrid
from datetime import datetime, timezone

df = climagrid.run(
    "my_transformers.csv",          # CSV with asset_id, lat, lon columns
    start_dt=datetime(2024, 7, 1,  tzinfo=timezone.utc),
    end_dt=datetime(2024, 7, 31, tzinfo=timezone.utc),
    sources=["nasa_power"],         # no API key required
    features="all",
)
df.to_parquet("stress_features.parquet")
```

![climagrid terminal demo](https://raw.githubusercontent.com/TemidireAdesiji/climagrid/main/docs/assets/demo.gif)

You now have a file you can join to your maintenance records and feed into whatever model or spreadsheet you already use. See the [quickstart notebook](https://github.com/TemidireAdesiji/climagrid/blob/main/examples/quickstart.ipynb) for a worked example.

Prefer a ready-to-read deliverable over a DataFrame? One command turns an asset list into a ranked inspection report (PDF) and an inspection-list CSV, no Python required:

```bash
pip install "climagrid[report]"
climagrid report --assets my_assets.csv --start 2024-07-01 --end 2024-07-31
```

Each asset is ranked by the single weather hazard it is most exposed to, with the driving stress and its standard named. It is a prioritization aid, not a failure prediction (see [Validation Notes](https://github.com/TemidireAdesiji/climagrid/blob/main/docs/validation_notes.md) for what climagrid does and does not establish).

![Asset thermal stress map, 33 real substations across 7 U.S. states](https://raw.githubusercontent.com/TemidireAdesiji/climagrid/main/docs/assets/quickstart_map.png)

For the lower-level adapter API (fetching individual data sources, custom feature computation) see the [documentation](https://climagrid.readthedocs.io).

---

## Data sources

| Agency | Dataset | Variables | Frequency |
|---|---|---|---|
| NOAA | HRRR (3 km CONUS) | Temperature, wind, precipitation, humidity, solar | Hourly |
| NOAA | NCEI CDO (stations) | Temperature, wind, precipitation | Hourly/Daily |
| NASA | POWER API (MERRA-2) | Surface meteorology, irradiance | Hourly/Daily |
| USDA NRCS | SCAN / SNOTEL | Soil moisture, soil temperature, snow water equivalent | Hourly |
| USFS / NIFC | WFIGS | Active wildfire perimeters, fire area | Daily |

All sources are free and publicly accessible. NOAA NCEI requires a free API token (register at ncdc.noaa.gov/cdo-web/token, then set the `NOAA_CDO_TOKEN` environment variable). All other sources work without credentials.

---

## Environmental stress features

| Feature | Output column | Target assets | Standard |
|---|---|---|---|
| Transformer thermal aging factor | `feat_thermal_aging_factor` | Transformers | IEEE C57.91 |
| Heat accumulation hours | `feat_heat_hours_above_35c` | Transformers, switchgear | IEEE C57.91 |
| Freeze-thaw cycles | `feat_freeze_thaw_cycles` | Conductors, insulators, poles | n/a |
| Ice loading risk | `feat_ice_loading_risk` | Overhead T&D lines | ASCE 7-22 (simplified) |
| Soil saturation index | `feat_soil_saturation_index` | Underground cables, poles | n/a |
| Wildfire proximity score | `feat_wildfire_proximity` | All overhead assets | n/a |
| Conductor sag index | `feat_conductor_sag_index` | Overhead T&D lines | IEEE 738 (simplified) |

---

## How this is different from existing tools

| Tool | What it does | How climagrid is different |
|---|---|---|
| atlite (PyPSA) | Turns weather into renewable-generation forecasts | climagrid focuses on environmental stress on equipment, not generation |
| ERAD (NREL) | Estimates damage from one-time disasters (hurricanes, floods) | climagrid produces continuous stress features for day-to-day maintenance |
| OpenSTEF | Forecasts feeder load using weather inputs | climagrid produces environmental stress features, not load forecasts |
| NRECA OMF | Simulates rural-coop feeders to evaluate new technologies | climagrid feeds maintenance models, not feeder simulators |
| EEweather (OpenDSM) | Pulls NOAA temperature for energy-efficiency baselines | climagrid integrates five federal sources and produces grid-stress features |

See the [full related-work review](https://github.com/TemidireAdesiji/climagrid/blob/main/docs/related_work.md).

---

## National impact

- About **42 million Americans** are served by rural electric cooperatives, covering **56% of U.S. landmass** (NRECA).
- Major U.S. power outages cost roughly **$67 billion per year** on average (Oak Ridge National Laboratory).
- Predictive maintenance, the broader practice, reduces equipment-maintenance costs by **25 to 30 percent** with about a **10-to-1 return on investment** (U.S. Department of Energy). climagrid's contribution is to make the underlying environmental stress data free and accessible, lowering the barrier for smaller utilities to adopt that practice.
- climagrid is released under the **Apache 2.0 license** so that any utility, regardless of size or budget, can use, modify, and redistribute it freely and permanently.

For more on how climagrid ties into U.S. grid resilience priorities, see [docs/national_impact.md](https://github.com/TemidireAdesiji/climagrid/blob/main/docs/national_impact.md).

---

## Citation

If you use climagrid in research, regulatory filings, or utility planning, please cite it using the metadata in [CITATION.cff](https://github.com/TemidireAdesiji/climagrid/blob/main/CITATION.cff), or:

```
Adesiji, T. (2026). climagrid: Open-source environmental stress feature toolkit
to support electric utility predictive maintenance (v0.2.1). Apache 2.0.
https://doi.org/10.5281/zenodo.20256535
```

---

## Contributing

climagrid welcomes contributions from utility engineers, data scientists, and researchers. See [CONTRIBUTING.md](https://github.com/TemidireAdesiji/climagrid/blob/main/CONTRIBUTING.md) for how to add a new data source adapter or stress feature. Project governance is documented in [GOVERNANCE.md](https://github.com/TemidireAdesiji/climagrid/blob/main/GOVERNANCE.md).

---

## License

Apache License 2.0. See [LICENSE](https://github.com/TemidireAdesiji/climagrid/blob/main/LICENSE) for full text.

Copyright 2026 Temidire Adesiji
