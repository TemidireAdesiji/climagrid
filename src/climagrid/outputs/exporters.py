"""
Export helpers — write a climagrid environmental DataFrame to various formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import pandas as pd

from climagrid.schema import COLUMN_MAP, ColumnSpec


def to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the DataFrame as-is after schema validation.

    Useful as a no-op endpoint that confirms the DataFrame conforms
    to the climagrid schema without writing to disk.
    """
    from climagrid.schema import validate_dataframe
    errors = validate_dataframe(df)
    if errors:
        import warnings
        warnings.warn(
            f"Schema validation warnings:\n" + "\n".join(f"  - {e}" for e in errors),
            UserWarning,
            stacklevel=2,
        )
    return df


def to_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> Path:
    """
    Write the DataFrame to a CSV file.

    Parameters
    ----------
    df:
        climagrid environmental DataFrame.
    path:
        Output file path (will create parent directories).
    **kwargs:
        Passed to pandas.DataFrame.to_csv().

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("index", False)
    df.to_csv(path, **kwargs)
    return path.resolve()


def to_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> Path:
    """
    Write the DataFrame to a Parquet file (column-oriented, compressed).

    Parquet is recommended for time ranges > 30 days or > 100 assets.

    Parameters
    ----------
    df:
        climagrid environmental DataFrame.
    path:
        Output file path.
    **kwargs:
        Passed to pandas.DataFrame.to_parquet().

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("index", False)
    kwargs.setdefault("compression", "snappy")
    df.to_parquet(path, **kwargs)
    return path.resolve()


def to_long_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> Path:
    """
    Write the DataFrame to long-form Parquet: one row per (asset_id, timestamp, feature_name).

    Long form is the preferred format for ML feature stores and streaming pipelines.
    Schema: asset_id | timestamp | lat | lon | feature_name | feature_value

    Parameters
    ----------
    df:
        Wide-form climagrid DataFrame (output of ``climagrid.run()``).
    path:
        Output file path.
    **kwargs:
        Passed to pandas.DataFrame.to_parquet().

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    id_cols = [c for c in ["asset_id", "timestamp", "lat", "lon"] if c in df.columns]
    value_cols = [c for c in df.columns if c not in id_cols]
    long = df.melt(id_vars=id_cols, var_name="feature_name", value_name="feature_value")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("index", False)
    kwargs.setdefault("compression", "snappy")
    long.to_parquet(path, **kwargs)
    return path.resolve()


def to_json_schema(path: Union[str, Path, None] = None) -> dict:
    """
    Return (and optionally write) the climagrid column schema as JSON.

    The schema file describes every column's name, dtype, units,
    source, and description. Utility engineers and SCADA vendors can
    use this to auto-configure their data ingestion pipelines.

    Parameters
    ----------
    path:
        If provided, write the schema JSON to this path.

    Returns
    -------
    dict
        The schema as a Python dictionary (also written to path if given).
    """
    schema: dict = {
        "climagrid_schema_version": "0.1.0",
        "description": (
            "Column schema for climagrid EnvironmentalDataFrame. "
            "All adapters and feature functions produce DataFrames "
            "conforming to this specification."
        ),
        "columns": [],
    }

    for spec in COLUMN_MAP.values():
        schema["columns"].append(
            {
                "name": spec.name,
                "dtype": spec.dtype,
                "units": spec.units,
                "source": spec.source,
                "nullable": spec.nullable,
                "description": spec.description,
            }
        )

    if path is not None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

    return schema
