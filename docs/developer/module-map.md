# Module Map

## CLI and Orchestration

- `photoelastimetry/main.py`
  - CLI entry points
  - parameter merging and alias handling
  - workflow orchestration

## Core Numerical and Physics Modules

- `photoelastimetry/image.py`
  - Mueller matrix operations
  - Stokes processing
  - forward simulation helpers
- `photoelastimetry/seeding.py`
  - phase-decomposed seeding
  - fringe-order resolution
- `photoelastimetry/optimise.py`
  - pressure/mean-stress recovery
- `photoelastimetry/bspline.py`
  - spline basis and projection utilities
- `photoelastimetry/unwrapping.py`
  - graph-cut angle unwrapping

## Data IO and Visualization

- `photoelastimetry/io.py`
  - raw/image load/save
  - Bayer pixel-format support
  - channel split and binning
- `photoelastimetry/plotting.py`
  - plotting and visualization helpers
- `photoelastimetry/visualisation.py`
  - boundary condition ASCII visualization

## Calibration

- `photoelastimetry/calibrate.py`
  - calibration config validation
  - blank correction
  - fitting and output generation

## Synthetic Generators

- `photoelastimetry/generate/disk.py`
- `photoelastimetry/generate/point_load.py`
- `photoelastimetry/generate/strip_load.py`
- `photoelastimetry/generate/lithostatic.py`
- `photoelastimetry/generate/inclined_plane.py`

## Test Coverage Map

- CLI and config behavior: `tests/test_main.py`
- Calibration: `tests/test_calibrate.py`
- IO: `tests/test_io_functions.py`
- Solver: `tests/test_optimise.py`
- Seeding/unwrapping: `tests/test_seeding.py`, `tests/test_unwrapping.py`
- B-spline internals: `tests/test_bspline.py`, `tests/test_bspline_analytical.py`
- Generator modules: `tests/test_disk.py`, `tests/test_point_load.py`, `tests/test_strip_load.py`, `tests/test_lithostatic.py`, `tests/test_inclined_plane.py`
