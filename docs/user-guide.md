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

- `p_filename`: Path to the photoelastimetry parameter file
- `stress_filename`: Path to the stress field data file
- `t`: Thickness of the photoelastic material
- `lambda_light`: Wavelength of light used in the experiment
- `C`: Stress-optic coefficient of the material
- `scattering` (optional): Gaussian filter sigma for scattering simulation
- `output_filename` (optional): Path for the output image (default: "output.png")

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

## Stress Analysis Methods

The package provides three complementary approaches for stress field recovery:

### Stokes-based Solver

Uses normalised Stokes components for pixel-wise stress inversion. This is deprecated and should not be used.

- Module: `photoelastimetry.optimiser.stokes`

### Intensity-based Solver

Works directly with raw polarisation intensities for pixel-wise inversion. Only used for validation purposes.

- Module: `photoelastimetry.optimiser.intensity`

### Equilibrium Solver

Global inversion that enforces mechanical equilibrium constraints using an Airy stress function. This is the recommended method for accurate stress recovery.

- Best for: Cases where mechanical equilibrium is important
- Module: `photoelastimetry.optimiser.equilibrium`

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
5. **Validation**: Compare results from different solver methods when possible
