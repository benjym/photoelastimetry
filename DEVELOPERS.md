# Development

To set up the development environment, clone the repository and install the package in editable mode with development dependencies:

```bash
git clone https://github.com/benjym/photoelastimetry.git
cd photoelastimetry
pip install -e ".[dev]"
# Set up pre-commit hooks
pre-commit install
```

## Running Tests

The project uses `pytest` for testing with comprehensive coverage analysis:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=photoelastimetry --cov-report=html

# Run specific test file
pytest tests/test_stokes_solver_pytest.py -v

# Run tests in parallel (faster)
pytest -n auto
```

## Code Coverage

View the coverage report by opening `htmlcov/index.html` in your browser after running tests with coverage enabled.

Current test coverage includes:
- Stokes solver: photoelastic stress recovery using normalised Stokes parameters
- Intensity solver: raw intensity-based stress recovery with noise modelling
- Equilibrium solver: global stress field recovery enforcing mechanical equilibrium
- Disk simulations: synthetic photoelastic data generation
- Image processing: retardance, principal angle, and Mueller matrix calculations

## Code Quality

The project uses `black` for code formatting and `flake8` for linting:

```bash
# Format code
black photoelastimetry tests

# Check code style
flake8 photoelastimetry
```

## Continuous Integration

GitHub Actions automatically runs tests on:
- Python 3.9, 3.10, 3.11, and 3.12
- Multiple operating systems (Ubuntu)
- Every push and pull request

Test coverage is automatically uploaded to Codecov for tracking.
