# Test Suite Documentation

## Overview

The photoelastimetry package uses `pytest` for unit and integration tests.

## Test Structure

```
tests/
├── __init__.py
├── test_optimise.py
├── test_main.py
├── test_seeding.py
├── test_disk.py
├── test_point_load.py
├── test_inclined_plane.py
├── test_elastic.py
├── test_bspline_analytical.py
├── test_image_io.py
├── test_io_functions.py
└── ...
```

## Core Coverage

### Optimise Solver (`test_optimise.py`)

- Invariant preservation of `(delta_sigma, theta)` in reconstructed stresses
- External potential handling
- Gauge behaviour when pressure boundary conditions are absent

### Main Pipeline (`test_main.py`)

- End-to-end `image_to_stress` path using optimise solver
- Config validation for removed keys
- Output writing, crop/binning, and CLI wrappers

### Seeding (`test_seeding.py`)

- Retardance inversion
- Fringe order resolution

## Running Tests

```bash
# Run all tests
pytest

# Run a specific file
pytest tests/test_optimise.py -v

# Coverage report
pytest --cov=photoelastimetry --cov-report=term-missing
```

## Continuous Integration

Tests run automatically on GitHub Actions across supported Python versions.
