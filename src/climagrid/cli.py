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
    """climagrid: climate data, grid-ready.

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
    from climagrid.outputs import to_csv, to_long_parquet, to_parquet

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
        click.secho("Warning: result is empty: check source availability.", fg="yellow")

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
    "--assets", "-a",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Asset CSV or GeoJSON file (must have asset_id, lat, lon columns).",
)
@click.option("--start", "-s", required=True, metavar="YYYY-MM-DD", help="Start date (UTC).")
@click.option("--end", "-e", required=True, metavar="YYYY-MM-DD", help="End date (UTC, inclusive).")
@click.option(
    "--sources", default="nasa_power", show_default=True,
    help="Comma-separated data source names.",
)
@click.option(
    "--features", default="all", show_default=True,
    help="Comma-separated feature names or 'all'.",
)
@click.option(
    "--output", "-o",
    default="climagrid_report.pdf",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output PDF path. A companion inspection-list CSV is written alongside it.",
)
@click.option(
    "--top-n", default=20, show_default=True,
    help="Number of highest-priority assets to list in the PDF.",
)
@click.option("--title", default=None, help="Report title (optional).")
@click.option("--bbox-radius", default=50.0, show_default=True, metavar="KM")
def report(
    assets: Path,
    start: str,
    end: str,
    sources: str,
    features: str,
    output: Path,
    top_n: int,
    title: str | None,
    bbox_radius: float,
) -> None:
    """Generate a co-op-ready PDF inspection report (priority list and map)."""
    import climagrid
    from climagrid.outputs import generate_report, rank_assets

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

    click.echo(f"Assets:  {assets}")
    click.echo(f"Period:  {start} to {end}")

    try:
        df = climagrid.run(
            assets, start_dt=start_dt, end_dt=end_dt,
            sources=source_list, features=feature_list, bbox_radius_km=bbox_radius,
        )
    except Exception as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        sys.exit(1)

    if df.empty:
        click.secho("Error: no data returned; cannot build a report.", fg="red", err=True)
        sys.exit(1)

    ranked = rank_assets(df)
    csv_path = output.with_name(output.stem + "_inspection_list.csv")
    ranked.to_csv(csv_path, index=False)

    try:
        pdf_path = generate_report(
            df, output,
            title=title or "climagrid asset stress report",
            top_n=top_n,
            period=f"{start} to {end}",
        )
    except ImportError as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        click.secho(f"The inspection-list CSV was still written to {csv_path}.", fg="yellow")
        sys.exit(1)

    top = ranked.head(min(top_n, 10))
    click.echo("\nHighest-priority assets:")
    for _, r in top.iterrows():
        click.echo(f"  {int(r['rank']):>3}. {str(r['asset_id']):<24} {r['dominant_hazard']}")
    click.secho(f"\n✓ PDF report:      {pdf_path}", fg="green")
    click.secho(f"✓ Inspection list: {csv_path}", fg="green")


@main.command()
@click.option(
    "--assets", "-a",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Asset CSV or GeoJSON file (must have asset_id, lat, lon columns).",
)
@click.option(
    "--history-start", default=None, metavar="YYYY-MM-DD",
    help="Start of training history (UTC). Default: derived from --history-years.",
)
@click.option(
    "--history-end", default=None, metavar="YYYY-MM-DD",
    help="End of training history (UTC). Default: today.",
)
@click.option(
    "--history-years", default=15, show_default=True,
    help="History length in years when --history-start is not given.",
)
@click.option(
    "--horizon", default=7, show_default=True,
    help="Forecast horizon in days.",
)
@click.option(
    "--targets", default="feat_thermal_aging_factor", show_default=True,
    help="Comma-separated stress-feature target columns to forecast.",
)
@click.option(
    "--model", default="lightgbm", show_default=True,
    type=click.Choice(["lightgbm", "persistence", "climatology"]),
    help="Forecaster: the trained model or a baseline.",
)
@click.option(
    "--sources", default="nasa_power", show_default=True,
    help="Comma-separated data source names.",
)
@click.option(
    "--backtest", is_flag=True, default=False,
    help="Emit a rolling-origin skill-score table instead of a forward forecast.",
)
@click.option(
    "--output", "-o",
    default="climagrid_forecast.parquet",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Output file path (.parquet or .csv).",
)
def forecast(
    assets: Path,
    history_start: str | None,
    history_end: str | None,
    history_years: int,
    horizon: int,
    targets: str,
    model: str,
    sources: str,
    backtest: bool,
    output: Path,
) -> None:
    """Forecast asset stress features ahead (probabilistic, standards-based)."""
    import pandas as pd

    import climagrid
    from climagrid.forecasting import ForecastConfig
    from climagrid.forecasting.backtest import evaluate
    from climagrid.forecasting.dataset import build_training_panel

    target_list = [t.strip() for t in targets.split(",") if t.strip()]
    source_list = [s.strip() for s in sources.split(",") if s.strip()]

    try:
        config = ForecastConfig(
            targets=target_list,
            horizon_days=horizon,
            model=model,  # type: ignore[arg-type]
            history_years=history_years,
            sources=source_list,
        )
    except Exception as exc:
        raise click.BadParameter(str(exc)) from exc

    try:
        end_dt = (
            datetime.fromisoformat(history_end).replace(tzinfo=timezone.utc)
            if history_end
            else datetime.now(timezone.utc)
        )
        if history_start:
            start_dt = datetime.fromisoformat(history_start).replace(
                tzinfo=timezone.utc
            )
        else:
            start_dt = (
                pd.Timestamp(end_dt) - pd.DateOffset(years=history_years)
            ).to_pydatetime()
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--history-start/--history-end") from exc

    click.echo(f"Assets:   {assets}")
    click.echo(f"History:  {start_dt.date()} to {end_dt.date()}")
    click.echo(f"Targets:  {', '.join(target_list)}")
    click.echo(f"Horizon:  {horizon} days   Model: {model}")

    try:
        if backtest:
            panel = build_training_panel(assets, start_dt, end_dt, config)
            if panel.empty:
                click.secho("Error: no data returned; cannot backtest.", fg="red", err=True)
                sys.exit(1)
            result = evaluate(panel, config)
        else:
            result = climagrid.forecast(
                assets, config=config, history_start=start_dt, history_end=end_dt
            )
    except ImportError as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        sys.exit(1)
    except Exception as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        sys.exit(1)

    if result.empty:
        click.secho("Warning: result is empty: check source availability.", fg="yellow")

    suffix = output.suffix.lower()
    if suffix == ".csv":
        result.to_csv(output, index=False)
        fmt = "CSV"
    else:
        result.to_parquet(output, index=False)
        fmt = "Parquet"

    click.secho(f"✓ {len(result):,} rows → {output} ({fmt})", fg="green")


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
