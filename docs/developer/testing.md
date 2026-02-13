# Testing

## Test Stack

- Test runner: `pytest`
- Coverage config: `pyproject.toml` (`fail_under = 70`)
- Test roots: `tests/test_*.py`
- Markers: `slow`, `integration`, `unit`, `smoke`

## Common Commands

```bash
# Full suite (includes coverage reporting via pytest addopts)
pytest

# Exclude smoke tests
pytest -m "not smoke"

# Smoke tests only
pytest -m smoke

# Single test file
pytest tests/test_main.py
```

## What Is Covered

- CLI behavior and parameter validation (`tests/test_main.py`)
- Calibration configuration and fitting (`tests/test_calibrate.py`)
- IO format handling and channel mapping (`tests/test_io_functions.py`)
- Seeding, unwrapping, B-spline, and optimise solver behavior
- Generator modules (`disk`, `point_load`, `strip_load`, `lithostatic`, `inclined_plane`)

## Quality Expectations

- Add tests for new functionality and failure paths.
- Prefer deterministic arrays and fixed RNG seeds.
- Assert behavior (not only shape checks).
