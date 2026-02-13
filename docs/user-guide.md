# User Guide

## Overview

Photoelastimetry is a package for processing polarised images to measure stress in granular media using photoelastic techniques. This guide covers the main workflows and configuration options.

## Command Line Tools

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

**JSON5 Parameters:**

The JSON5 parameter file should contain:

- `folderName`: Path to folder containing raw photoelastic images
- `C`: Stress-optic coefficient in 1/Pa
- `thickness`: Sample thickness in meters
- `wavelengths`: List of wavelengths in nanometers
- `S_i_hat`: Incoming normalised Stokes vector [S1_hat, S2_hat, S3_hat] representing polarisation state
- `crop` (optional): Crop region as [y1, y2, x1, x2]
- `debug` (optional): If true, display all channels for debugging
- `seeding` (optional): Controls phase-decomposed seeding (`enabled`, `n_max`, `sigma_max`)
- `knot_spacing`, `spline_degree`, `boundary_mask_file`, `boundary_values_files`, `boundary_weight`, `regularisation_weight` (`regularization_weight` alias), `regularisation_order`, `external_potential_file`, `external_potential_gradient`, `max_iterations`, `tolerance`, `verbose` (optional): Optimise solver settings

**Example parameter file:**

```json
{
  "folderName": "./images/experiment1",
  "C": 5e-11,
  "thickness": 0.005,
  "wavelengths": [450, 550, 650],
  "S_i_hat": [1.0, 0.0, 0.0],
  "crop": [100, 900, 100, 900],
  "debug": false
}
```

### stress-to-image

Converts stress field data to photoelastic fringe pattern images. This is useful for validating stress field calculations or generating synthetic training data.

```bash
stress-to-image <json_filename>
```

**Arguments:**

- `json_filename`: Path to the JSON5 parameter file containing configuration (required)

**Example:**

```bash
stress-to-image params.json5
```

**JSON5 Parameters:**

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
demosaic-raw <input_file> [OPTIONS]
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

## Stress Analysis Method

The package now uses a single solver path:

### Optimise Solver

Global pressure recovery that keeps seeded deviatoric stress fixed and solves only for mean stress with equilibrium and optional boundary/potential constraints.

- Best for: Workflows that trust seeding for `(σ1-σ2, θ)` and want stable pressure recovery
- Module: `photoelastimetry.optimise`
- Note: `image-to-stress` always uses this solver; `solver`, `global_solver`, and `global_mean_stress` config keys are no longer supported.

## Photoelastic Theory

### Stress-Optic Law

The fundamental relationship between stress and optical retardation:

```
δ = C · t · (σ₁ - σ₂)
```

Where:

- δ is the optical retardation
- C is the stress-optic coefficient
- t is the specimen thickness
- σ₁, σ₂ are the principal stresses

### Mueller Matrix Formalism

The package uses Mueller matrix calculus to model light propagation through the photoelastic sample and optical elements.

## Tips and Best Practices

1. **Calibration**: Always calibrate the stress-optic coefficient (C) for your specific material
2. **Image Quality**: Use high-quality, well-exposed images with minimal noise
3. **Wavelength Selection**: Multiple wavelengths improve stress field resolution
4. **Cropping**: Crop images to regions of interest to reduce computation time
5. **Validation**: Compare against known analytical/synthetic solutions when possible
