from climagrid.sources.base import BaseEnvironmentalSource, BoundingBox
from climagrid.sources.nasa_power import NasaPowerAdapter
from climagrid.sources.noaa_ncei import NceiAdapter
from climagrid.sources.usda_nrcs import NrcsAdapter
from climagrid.sources.usfs_wfigs import WfigsAdapter

__all__ = [
    "BaseEnvironmentalSource",
    "BoundingBox",
    "NasaPowerAdapter",
    "NceiAdapter",
    "NrcsAdapter",
    "WfigsAdapter",
]

# HrrrAdapter requires optional herbie-data dependency
try:
    from climagrid.sources.noaa_hrrr import HrrrAdapter  # noqa: F401
    __all__.append("HrrrAdapter")
except ImportError:
    pass
