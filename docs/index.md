# climagrid

**Climate data, grid-ready.**

climagrid is an open-source Python toolkit that converts public NOAA, NASA, USDA, and U.S. Forest Service environmental data into standardized predictive-maintenance input features for electric utility grid resilience systems.

Built for rural electric cooperatives and municipal utilities serving approximately 42 million Americans in underserved and high-risk service territories.

```{code-block} python
import climagrid
from datetime import datetime, timezone

df = climagrid.run(
    "my_coop_assets.csv",
    start_dt=datetime(2024, 7, 1, tzinfo=timezone.utc),
    end_dt=datetime(2024, 7, 8, tzinfo=timezone.utc),
    sources=["nasa_power"],
    features="all",
)
# → DataFrame: asset_id, timestamp, nasa_temperature_2m, ...,
#              feat_thermal_aging_factor, feat_conductor_sag_index, ...
```

---

## Why climagrid?

Electric utilities, especially small rural cooperatives, need high-quality environmental stress data to predict equipment failures before they cascade into outages. The raw inputs exist across several free government APIs, but translating them into ML-ready features requires domain knowledge spanning meteorology, power systems engineering, and geospatial data processing.

climagrid handles that translation in one place, under an Apache 2.0 license, with no API keys required for the default data sources.

---

## Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

getting_started
data_sources
schema
```

```{toctree}
:maxdepth: 2
:caption: Context

related_work
national_impact
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

---

## License

Apache 2.0: free for commercial and research use, patent grants included.
See [LICENSE](https://github.com/TemidireAdesiji/climagrid/blob/main/LICENSE).
