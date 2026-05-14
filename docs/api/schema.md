# Schema

## ColumnSpec

```{eval-rst}
.. autoclass:: climagrid.schema.ColumnSpec
   :members:
```

## Utility functions

```{eval-rst}
.. autofunction:: climagrid.schema.validate_dataframe

.. autofunction:: climagrid.schema.schema_summary

.. autofunction:: climagrid.schema.empty_dataframe
```

## Column lists

The following module-level lists are exported for programmatic use:

```python
from climagrid.schema import (
    ALL_COLUMNS,       # all 34 ColumnSpec objects
    COLUMN_MAP,        # dict[name, ColumnSpec]
    INDEX_COLUMNS,
    NOAA_HRRR_COLUMNS,
    NASA_POWER_COLUMNS,
    NOAA_NCEI_COLUMNS,
    USDA_NRCS_COLUMNS,
    USFS_WFIGS_COLUMNS,
    FEATURE_COLUMNS,
)
```
