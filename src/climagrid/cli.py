"""
Command-line interface for climagrid.

Usage examples
--------------
  climagrid fetch --assets assets.csv --start 2024-07-01 --end 2024-07-08
  climagrid fetch --assets assets.csv --start 2024-07-01 --end 2024-07-08 \\
      --sources nasa_power,usfs_wfigs --output features.parquet
  climagrid schema
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import click


@click.group()
@click.version_option(package_name="climagrid")
def main() -> None:
    """climagrid — climate data, grid-ready.

    Fetch NOAA/NASA/USDA/USFS environmental data and compute
    predictive-maintenance stress features for utility assets.
    """


@main.command()
@click.option(
    "--assets", "-a",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Asset CSV or GeoJSON file (must have asset_id, lat, lon columns).",
)
@click.option(
    "--start", "-s",
    required=True,
    metavar="YYYY-MM-DD",
    help="Start date (UTC).",
)
@click.option(
    "--end", "-e",
    required=True,
    metavar="YYYY-MM-DD",
    help="End date (UTC, inclusive).",
)
@click.option(
    "--sources",
    default="nasa_power",
    show_default=True,
    help="Comma-separated data source names. "
         "Valid: nasa_power, noaa_hrrr, noaa_ncei, usda_nrcs, usfs_wfigs.",
)
@click.option(
    "--features",
    default="all",
    show_default=True,
    help="Comma-separated feature names or 'all'. "
         "Valid: thermal, conductor_sag, freeze_thaw, ice_loading, soil, wildfire.",
)
@click.option(
    "--output", "-o",
    default="climagrid_output.parquet",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output file path (.parquet or .csv).",
)
@click.option(
    "--long-form",
    is_flag=True,
    default=False,
    help="Write long-form Parquet (feature_name, feature_value rows) instead of wide.",
)
@click.option(
    "--bbox-radius",
    default=50.0,
    show_default=True,
    metavar="KM",
    help="Bounding box radius around asset centroid for data fetch.",
)
def fetch(
    assets: Path,
    start: str,
    end: str,
    sources: str,
    features: str,
    output: Path,
    long_form: bool,
    bbox_radius: float,
) -> None:
    """Fetch environmental data and compute stress features for utility assets."""
    import climagrid
    from climagrid.outputs import to_parquet, to_csv, to_long_parquet

    try:
        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--start/--end") from exc

    source_list = [s.strip() for s in sources.split(",") if s.strip()]
    feature_list: list[str] | str = (
        "all"
        if features.strip().lower() == "all"
        else [f.strip() for f in features.split(",") if f.strip()]
    )

    click.echo(f"Assets:   {assets}")
    click.echo(f"Period:   {start} → {end}")
    click.echo(f"Sources:  {', '.join(source_list)}")
    click.echo(f"Features: {features}")

    try:
        df = climagrid.run(
            assets,
            start_dt=start_dt,
            end_dt=end_dt,
            sources=source_list,
            features=feature_list,
            bbox_radius_km=bbox_radius,
        )
    except Exception as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        sys.exit(1)

    if df.empty:
        click.secho("Warning: result is empty — check source availability.", fg="yellow")

    suffix = output.suffix.lower()
    if long_form and suffix == ".parquet":
        out_path = to_long_parquet(df, output)
        fmt = "long-form Parquet"
    elif suffix == ".csv":
        out_path = to_csv(df, output)
        fmt = "CSV"
    else:
        out_path = to_parquet(df, output)
        fmt = "Parquet"

    click.secho(
        f"✓ {len(df):,} rows × {df.shape[1]} columns → {out_path} ({fmt})",
        fg="green",
    )


@main.command()
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional path to write schema JSON file.",
)
def schema(output: Path | None) -> None:
    """Print the climagrid column schema."""
    import climagrid

    summary = climagrid.schema_summary()
    click.echo(summary.to_string(index=False))

    if output:
        from climagrid.outputs import to_json_schema
        to_json_schema(output)
        click.secho(f"✓ Schema written to {output}", fg="green")
