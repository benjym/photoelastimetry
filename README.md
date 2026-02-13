# photoelastimetry

[![Tests](https://github.com/benjym/photoelastimetry/actions/workflows/test.yml/badge.svg)](https://github.com/benjym/photoelastimetry/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/benjym/photoelastimetry/branch/main/graph/badge.svg)](https://codecov.io/gh/benjym/photoelastimetry)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Package for processing polarised images to measure stress in granular media

## Installation

To install the package, run the following command in the terminal:

```bash
pip install photoelastimetry
```

## Documentation

Full documentation is available [here](https://benjym.github.io/photoelastimetry/).

## Usage

After installation, two command line scripts are available:

### image-to-stress

Converts photoelastic images to stress maps using the stress-optic law and polarisation analysis.

```bash
image-to-stress <json_filename> [--output OUTPUT]
```

**Arguments:**

- `json_filename`: Path to the JSON5 parameter file containing configuration (required)
- `--output`: Path to save the output stress map image (optional)

**Example:**

```bash
image-to-stress params.json5 --output stress_map.png
```

The JSON5 parameter file should contain:

- `folderName`: Path to folder containing raw photoelastic images
- `C`: Stress-optic coefficient in 1/Pa
- `thickness`: Sample thickness in meters
- `wavelengths`: List of wavelengths in nanometers
- `S_i_hat`: Incoming normalised Stokes vector [S1_hat, S2_hat, S3_hat] representing polarisation state
- `crop` (optional): Crop region as [y1, y2, x1, x2]
- `debug` (optional): If true, display all channels for debugging
- `solver` (optional): One of `stokes`, `intensity`, `global`, `global_mean_stress` (default)

### stress-to-image

Converts stress field data to photoelastic fringe pattern images.

```bash
stress-to-image <json_filename>
```

**Arguments:**

- `json_filename`: Path to the JSON5 parameter file containing configuration (required)

**Example:**

```bash
stress-to-image params.json5
```

The JSON5 parameter file should contain:

- `stress_filename`: Path to the stress field data file (or legacy `s_filename`)
- `thickness` (or legacy `t`): Thickness of the photoelastic material
- `wavelengths` (nm or m) or legacy `lambda_light`: Illumination wavelengths
- `C`: Stress-optic coefficient(s), scalar or one per wavelength
- `S_i_hat` (optional): Incoming normalised Stokes vector `[S1_hat, S2_hat, S3_hat]`
- `stress_order` (optional): `xx_yy_xy` (default) or legacy `xy_yy_xx`
- `scattering` (optional): Gaussian filter sigma for scattering simulation
- `output_filename` (optional):
  - `.tiff/.tif/.npy/.raw`: saves full synthetic stack `[H, W, n_wavelengths, 4]`
  - `.png/.jpg/.jpeg`: saves a 2-panel fringe/isoclinic plot (default: `output.png`)

### demosaic-raw

De-mosaics a raw polarimetric image from a camera with a 4x4 superpixel pattern into separate colour and polarisation channels.

```bash
demosaic-raw <input_file> [--width WIDTH] [--height HEIGHT] [--dtype DTYPE] [--output-prefix PREFIX] [--format FORMAT] [--all]
```

**Arguments:**

- `input_file`: Path to the raw image file, or directory when using `--all` (required)
- `--width`: Image width in pixels (default: 4096)
- `--height`: Image height in pixels (default: 3000)
- `--dtype`: Data type, either 'uint8' or 'uint16' (auto-detected if not specified)
- `--output-prefix`: Prefix for output files (default: input filename without extension)
- `--format`: Output format, either 'tiff' or 'png' (default: 'tiff')
- `--all`: Recursively process all .raw files in the input directory and subdirectories

**Examples:**

```bash
# Save as a single TIFF stack
demosaic-raw image.raw --width 2448 --height 2048 --dtype uint16 --format tiff

# Save as four separate PNG files (one per polarisation angle)
demosaic-raw image.raw --width 2448 --height 2048 --format png --output-prefix output

# Process all raw files in a directory recursively
demosaic-raw images/ --format png --all
```

**Output formats:**

- `tiff`: Creates a single TIFF file with shape [H/4, W/4, 4, 4] containing all colour channels (R, G1, G2, B) and polarisation angles (0°, 45°, 90°, 135°)
- `png`: Creates 4 PNG files (one per polarisation angle), each containing all colour channels as a composite image

## Development

To set up the development environment, clone the repository and install the package in editable mode with development dependencies:

```bash
git clone https://github.com/benjym/photoelastimetry.git
cd photoelastimetry
pip install -e ".[dev]"
# Set up pre-commit hooks
pre-commit install
```

### Running Tests

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

### Code Coverage

View the coverage report by opening `htmlcov/index.html` in your browser after running tests with coverage enabled.

Current test coverage includes:

- Stokes solver: photoelastic stress recovery using normalised Stokes parameters
- Intensity solver: raw intensity-based stress recovery with noise modelling
- Equilibrium solver: global stress field recovery enforcing mechanical equilibrium
- Disk simulations: synthetic photoelastic data generation
- Image processing: retardance, principal angle, and Mueller matrix calculations

### Code Quality

The project uses `black` for code formatting and `flake8` for linting:

```bash
# Format code
black photoelastimetry tests

# Check code style
flake8 photoelastimetry
```

### Continuous Integration

GitHub Actions automatically runs tests on:

- Python 3.9, 3.10, 3.11, and 3.12
- Multiple operating systems (Ubuntu)
- Every push and pull request

Test coverage is automatically uploaded to Codecov for tracking.

## Authors

- [Benjy Marks](mailto:benjy.marks@sydney.edu.au)
