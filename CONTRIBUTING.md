# Contributing to climagrid

Thank you for helping improve grid resilience tooling for rural electric cooperatives. This document covers everything you need to get a change from idea to merged PR.

---

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Ways to contribute](#ways-to-contribute)
- [Local development setup](#local-development-setup)
- [Branch naming](#branch-naming)
- [Making a change](#making-a-change)
- [Pull request process](#pull-request-process)
- [Coding standards](#coding-standards)
- [Adding a new data source adapter](#adding-a-new-data-source-adapter)
- [Adding a new stress feature](#adding-a-new-stress-feature)

---

## Code of conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to abide by its terms. Report violations to temidireadesiji@gmail.com.

---

## Ways to contribute

- **Bug reports**: open an issue using the bug report template
- **Feature requests**: open an issue using the feature request template
- **Documentation fixes**: typos, unclear wording, missing examples
- **New source adapters**: additional NOAA, NASA, USDA, or other public environmental data sources
- **New feature functions**: new grid stress features backed by a published standard or methodology

If you are unsure whether a change is in scope, open a discussion issue before writing code.

---

## Local development setup

**Requirements:** Python 3.10+, Git

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/climagrid.git
cd climagrid

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install in editable mode with all dev dependencies
pip install -e ".[dev]"

# 4. Confirm the test suite passes before making any changes
pytest tests/ -m "not integration" -q
```

---

## Branch naming

| Type | Pattern | Example |
|---|---|---|
| Bug fix | `fix/<short-description>` | `fix/nasa-fill-value-nan` |
| New feature | `feat/<short-description>` | `feat/noaa-rap-adapter` |
| Documentation | `docs/<short-description>` | `docs/quickstart-notebook` |
| Refactor | `refactor/<short-description>` | `refactor/orchestrator-cache` |
| Tests | `test/<short-description>` | `test/wfigs-centroid-edge-cases` |

Always branch from `main`:

```bash
git checkout main
git pull origin main
git checkout -b fix/your-description
```

---

## Making a change

1. Make your changes in small, focused commits
2. Run linting and tests before pushing:

```bash
ruff check src/ tests/        # must be zero errors
pytest tests/ -m "not integration" -q
```

3. Push your branch and open a PR against `main`

---

## Pull request process

- Fill in the PR template completely: incomplete PRs will be asked to update before review
- Keep PRs focused: one logical change per PR
- All CI checks must pass before a PR can be merged
- At least one approving review is required
- PRs are merged via **squash merge**: your branch history is collapsed into one commit on `main`
- A maintainer will review within 5 business days

By submitting a PR you agree your contributions will be licensed under the Apache 2.0 License.

---

## Coding standards

- **Linter:** `ruff check src/ tests/` must pass with zero errors (enforced in CI)
- **Type hints:** all public functions and methods must have type annotations
- **Docstrings:** one-line summary for public classes and functions; skip for internal helpers
- **Comments:** only when the *why* is non-obvious: do not describe what the code does
- **No new dependencies** without prior discussion: climagrid intentionally keeps its dependency surface small
- **No live API calls in unit tests**: use the `responses` library for HTTP mocking

---

## Adding a new data source adapter

1. Create `src/climagrid/sources/<source_name>.py`
2. Subclass `BaseEnvironmentalSource` from `climagrid.sources.base`
3. Implement `source_name` (property) and `fetch(bbox, start_dt, end_dt)`: returns a DataFrame
4. Name columns using the `source_variable_unit` convention (e.g. `yourapi_temperature_2m`)
5. Add column definitions to `src/climagrid/schema.py`
6. Add `"your_source"` to `_SOURCE_MAP` in `src/climagrid/pipeline/orchestrator.py`
7. Register the adapter in `src/climagrid/sources/__init__.py`
8. Add unit tests in `tests/test_sources/test_<source_name>.py` with mocked HTTP responses
9. Add an integration test marked `@pytest.mark.integration`

Follow `src/climagrid/sources/nasa_power.py` as the reference implementation.

---

## Adding a new stress feature

1. Create `src/climagrid/features/your_feature.py` with a class that has a `.compute(df) -> pd.DataFrame` method
2. The method must accept arbitrary extra columns and return a copy with the new `feat_` column added
3. Add the column spec to `FEATURE_COLUMNS` in `schema.py`
4. Add it to `_FEATURE_MAP` in `orchestrator.py`
5. Include a reference to the engineering standard or peer-reviewed source the formula is drawn from
