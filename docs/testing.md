# Test Suite Documentation

## Overview

The project uses `pytest` with `pytest-cov` for deterministic, behavior-driven validation.

## Coverage Policy

- Core coverage gate: `>= 70%`
- Coverage is focused on computational and pipeline modules.
- Plotting and terminal-visualisation utilities are exercised by smoke tests but excluded from strict gating.

## Test Quality Rules

- Do not swallow runtime failures with broad `try/except` and `pytest.skip`.
- Do not duplicate suites across files.
- Use deterministic inputs (fixed seeds or explicit arrays).
- Every test must assert at least one behavioral invariant, not just shape/finite checks.

## Running Tests

```bash
# Full suite (core + smoke)
pytest

# Core-only checks used for coverage gating
pytest -m "not smoke"

# Smoke-only checks
pytest -m smoke --override-ini='addopts=-v --tb=short --strict-markers --strict-config'
```

## Continuous Integration

- Matrix core tests run on Python 3.9-3.12 with coverage gating.
- A separate smoke job runs plotting/visualisation smoke tests.
