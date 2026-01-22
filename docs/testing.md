# Test Suite Documentation

## Overview

The photoelastimetry package includes a comprehensive test suite using `pytest` to ensure code quality and correctness. Tests are organized by module and cover unit tests, integration tests, and edge cases.

## Test Structure

```
tests/
├── __init__.py                       # Test package initialization
├── test_stokes_solver_pytest.py      # Stokes-based stress recovery tests
├── test_intensity_solver.py          # Intensity-based stress recovery tests
├── test_global_solver.py             # Global stress recovery tests
├── test_disk.py                      # Disk simulation and synthetic data tests
├── test_image_io.py                  # Image processing and I/O tests
└── test_stokes_solver.py             # Legacy comprehensive tests (kept for reference)
```

## Test Coverage by Module

### Stokes Solver (`test_stokes_solver_pytest.py`)

Tests for the Stokes-based photoelastic stress recovery:

- **Stokes Components**: Computation of S0, S1, S2 from four-step polarimetry
- **Normalised Stokes**: Normalisation and edge case handling (zero S0)
- **Retardance**: Computation from stress tensor using stress-optic law
- **Principal Angle**: Determination of stress orientation
- **Mueller Matrix**: Wave plate birefringence modeling
- **Forward Model**: Predicting Stokes parameters from known stress
- **Stress Recovery**: Inverse problem solving for stress tensor
- **Solid Fraction**: Light attenuation through granular media
- **Stress Mapping**: Full-field stress recovery from image stacks

**Key Features Tested:**

- Multi-wavelength RGB approach
- Uniaxial and biaxial stress states
- Shear stress recovery
- Principal stress difference accuracy

### Intensity Solver (`test_intensity_solver.py`)

Tests for raw intensity-based stress recovery:

- **Intensity Prediction**: Forward model from stress to intensity
- **Residual Computation**: Optimization objective function
- **Stress Recovery**: Least-squares fitting with noise modeling
- **Weighted Recovery**: Measurement confidence weighting
- **Bounded Optimization**: Constrained parameter estimation
- **Stress Mapping**: Full-field intensity-based recovery
- **Stokes Comparison**: Validation against Stokes method

**Key Features Tested:**

- Four-step polarimetry (0°, 45°, 90°, 135°)
- Multi-wavelength RGB measurements
- Optimization methods (Levenberg-Marquardt, Trust Region)
- Initial guess sensitivity
- Bounds and constraints

### Disk Simulations (`test_disk.py`)

Tests for synthetic photoelastic data generation:

- **Four-Step Polarimetry**: Mueller calculus-based simulation
- **Wavelength Dependence**: Dispersion effects
- **Stress Dependence**: Intensity variation with stress magnitude
- **Synthetic Disk Data**: Full disk-under-compression simulation
- **Stress Distribution**: Expected patterns and symmetries
- **Parameter Validation**: Input checking and error handling

**Key Features Tested:**

- Circular disk under diametral compression
- Brazilian disk test geometry
- Theoretical stress field validation
- Noise addition for realistic data
- Multiple wavelengths and polarisation states

### Image Processing and I/O (`test_image_io.py`)

Tests for image processing functions:

- **Retardance**: Array input handling
- **Principal Angle**: Special cases (pure shear, no shear)
- **Mueller Matrix**: Identity and symmetry properties
- **Zero Stress**: Degenerate cases
- **Extreme Values**: Numerical stability
- **File I/O**: Loading and saving results
- **Configuration**: JSON/JSON5 parameter files

**Key Features Tested:**

- Vectorized operations on arrays
- Edge case handling (zero, inf, nan)
- File format support
- Error handling for missing/invalid files

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_stokes_solver_pytest.py

# Run specific test class
pytest tests/test_stokes_solver_pytest.py::TestStokesComponents

# Run specific test function
pytest tests/test_stokes_solver_pytest.py::TestStokesComponents::test_compute_stokes_components

# Verbose output
pytest -v

# Show print statements
pytest -s
```

### Coverage Analysis

```bash
# Generate coverage report
pytest --cov=photoelastimetry

# Generate HTML coverage report
pytest --cov=photoelastimetry --cov-report=html

# Generate XML coverage report (for CI)
pytest --cov=photoelastimetry --cov-report=xml

# Show missing lines
pytest --cov=photoelastimetry --cov-report=term-missing
```

### Test Selection

```bash
# Run only fast tests (exclude slow)
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests matching pattern
pytest -k "stokes"

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

## Test Fixtures

Common fixtures used across tests:

- `test_parameters`: Standard material and measurement parameters
  - Wavelengths (R, G, B)
  - Stress-optic coefficients
  - Sample thickness
  - Solid fraction
  - Incident polarisation state

- `sample_stress`: Representative stress tensor components
  - σ_xx, σ_yy, σ_xy values

- `sample_intensities`: Four-step polarimetry measurements
  - I_0°, I_45°, I_90°, I_135°

- `sample_grid`: Grid parameters for field-based tests
  - Height, width, shape

- `temp_directory`: Temporary directory for I/O tests

## Continuous Integration

Tests run automatically on GitHub Actions for:

- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **On events**: Push to main/develop, pull requests
- **Checks**:
  - All tests pass
  - Code formatting (black)
  - Linting (flake8)
  - Minimum 70% coverage

Coverage reports are automatically uploaded to Codecov.

## Writing New Tests

When adding new functionality, include tests that cover:

1. **Happy path**: Normal, expected usage
2. **Edge cases**: Boundary conditions, empty inputs, zeros
3. **Error handling**: Invalid inputs, exceptions
4. **Integration**: How the feature works with existing code
5. **Performance**: For computationally intensive operations (mark as `@pytest.mark.slow`)

Example test structure:

```python
import pytest
import numpy as np

@pytest.fixture
def sample_data():
    return {'x': 1.0, 'y': 2.0}

class TestNewFeature:
    """Tests for new feature X."""

    def test_basic_functionality(self, sample_data):
        """Test normal operation."""
        result = new_function(**sample_data)
        assert result > 0
        assert np.isfinite(result)

    def test_edge_case_zero(self):
        """Test with zero input."""
        result = new_function(x=0, y=0)
        assert result == 0

    def test_invalid_input(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            new_function(x=-1, y=2)
```

## Coverage Goals

Target coverage by module:

- **Core solvers**: >80% (stokes_solver, intensity_solver, global_solver)
- **Utilities**: >70% (image, io, plotting)
- **Applications**: >60% (disk, main)
- **Overall**: >70%

## Known Test Limitations

Some aspects are not fully tested:

- GUI/interactive features
- Hardware-specific I/O (camera interfaces)
- Large-scale performance tests
- Full integration with external dependencies

These are acceptable given the scientific computing focus and availability of manual validation procedures.

## Test Maintenance

- Review and update tests when APIs change
- Add tests for bug fixes (regression tests)
- Remove or mark obsolete tests
- Keep test execution time reasonable (<5 minutes for full suite)
- Update this documentation when test structure changes
