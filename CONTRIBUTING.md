# Contributing to climagrid

Thank you for helping improve grid resilience tooling for rural electric cooperatives.

## Quick start

```bash
git clone https://github.com/TemidireAdesiji/climagrid.git
cd climagrid
uv pip install -e ".[dev]"
uv run pytest tests/ -m "not integration" -q
```

## Adding a new data source

1. Create `src/climagrid/sources/your_source.py` subclassing `BaseEnvironmentalSource`.
2. Name columns using the `source_variable_unit` convention (e.g. `yourapi_temperature_2m`).
3. Add column definitions to `src/climagrid/schema.py`.
4. Add `"your_source"` to `_SOURCE_MAP` in `src/climagrid/pipeline/orchestrator.py`.
5. Write unit tests with `responses` mocking. No live API calls in `tests/`.

## Adding a new stress feature

1. Create `src/climagrid/features/your_feature.py` with a class that has a `.compute(df) -> pd.DataFrame` method.
2. The method must accept arbitrary extra columns and return a copy with the new `feat_` column added.
3. Add the column spec to `FEATURE_COLUMNS` in `schema.py`.
4. Add it to `_FEATURE_MAP` in `orchestrator.py`.

## Code style

- `ruff check src/ tests/` must pass (enforced in CI).
- No live API calls in unit tests. Use the `responses` library for HTTP mocking.
- All new columns must be defined in `schema.py` before being emitted.

## Pull request process

1. Fork and branch from `main`.
2. Run `uv run pytest tests/ -m "not integration"` locally.
3. Open a PR against `main` with a clear description of the change.
4. A maintainer will review within 5 business days.

## Reporting bugs

Open an issue at https://github.com/TemidireAdesiji/climagrid/issues with:
- Python version and OS
- Minimal reproducible example
- Full traceback

## License

By contributing, you agree your contributions will be licensed under the Apache 2.0 License.
