## Description

<!-- What does this PR do and why? Link to any related issue: "Fixes #123" -->

## Type of change

- [ ] Bug fix
- [ ] New feature (source adapter or stress feature)
- [ ] Documentation update
- [ ] Refactor (no behavior change)
- [ ] Test improvement

## Testing done

<!-- Describe how you tested this change -->

- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `pytest tests/ -m "not integration" -q` passes locally
- [ ] New code has corresponding unit tests

## Checklist

- [ ] Branch is based on `main` (not another feature branch)
- [ ] PR is focused on one logical change
- [ ] New columns (if any) are defined in `schema.py`
- [ ] Public functions have type annotations and a one-line docstring
- [ ] No live API calls in unit tests (use `responses` for HTTP mocking)
