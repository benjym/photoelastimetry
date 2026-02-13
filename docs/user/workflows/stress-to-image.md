# Workflow: stress-to-image

Generate synthetic polarimetric image data from a stress map.

## Prerequisites

- Stress map file with shape `[H, W, 3]`
- Material/optics parameters:
  - `thickness`
  - `wavelengths` (or legacy `lambda_light`)
  - `C` (scalar or per-wavelength)
  - optional `S_i_hat`

## Minimal Config

```json
{
  stress_filename: "stress.npy",
  thickness: 0.01,
  wavelengths: [650, 550, 450],
  C: [3e-9, 3e-9, 3e-9],
  S_i_hat: [1.0, 0.0, 0.0],
  output_filename: "synthetic_stack.tiff"
}
```

## Command

```bash
stress-to-image params.json5
```

## Output Modes

- Stack output (`.tiff/.tif/.npy/.raw`): full `[H, W, n_wavelengths, 4]` synthetic data
- Plot output (`.png/.jpg/.jpeg`): 2-panel fringe/isoclinic visual

Example plot output config:

```json
{
  stress_filename: "stress.npy",
  thickness: 0.01,
  wavelengths: [550],
  C: 3e-9,
  output_filename: "fringe.png"
}
```

## Common Failure Modes

- `Missing stress map path`:
  - provide `stress_filename` (or legacy `s_filename`)
- `Missing wavelengths`:
  - provide `wavelengths` or legacy `lambda_light`
- `C must be scalar or length N`:
  - align `C` length with wavelength count
- `Unsupported output extension`:
  - use one of the supported extensions above
- `S_i_hat must have length 2 or 3`:
  - pass `[S1_hat, S2_hat]` or `[S1_hat, S2_hat, S3_hat]`
