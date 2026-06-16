"""
climagrid: Climate data, grid-ready.

Open-source Python toolkit that converts public NOAA, NASA, USDA, and
U.S. Forest Service data into standardized environmental stress features
for electric utility predictive maintenance systems.

Designed for rural electric cooperatives and municipal utilities.

License: Apache 2.0
"""

from climagrid.forecasting import ForecastConfig, forecast
from climagrid.pipeline.orchestrator import run
from climagrid.schema import schema_summary, validate_dataframe

__version__ = "0.2.1"
__all__ = [
    "ForecastConfig",
    "forecast",
    "run",
    "schema_summary",
    "validate_dataframe",
]
