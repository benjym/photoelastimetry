# User Documentation

This area is organized by tasks, not by module internals.

## Workflow Chooser

- Start from raw camera `.raw` frames:
  - [demosaic-raw workflow](workflows/demosaic-raw.md)
  - then [image-to-stress workflow](workflows/image-to-stress.md)
- Start from an existing image stack (`.tiff`/`.npy`):
  - [image-to-stress workflow](workflows/image-to-stress.md)
- Start from an existing stress map (`[H, W, 3]`):
  - [stress-to-image workflow](workflows/stress-to-image.md)
- Need material/optics calibration first:
  - [calibration workflow](workflows/calibration.md)

## Core Pages

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Configuration Reference](configuration.md)
- [Python API Workflow](python-api.md)
- [Troubleshooting](troubleshooting.md)

## Data Shape Conventions

- Stress map: `[H, W, 3]` in order `[sigma_xx, sigma_yy, sigma_xy]` unless you set a legacy stress order.
- Polarimetric stack: `[H, W, n_wavelengths, 4]` where the last axis is analyzer angles `[0, 45, 90, 135]`.
