# Workflow: image-to-stress

Convert polarimetric image data into a stress map.

## Prerequisites

- Input data as either:
  - raw capture folder (`folderName`) containing `recordingMetadata.json` and `0000000/frame*.raw`, or
  - prepared image stack file (`input_filename`) with shape `[H, W, n_wavelengths, 4]`
- Material/optics parameters provided directly (`C`, `thickness`, `wavelengths`, `S_i_hat`) or through `calibration_file`

## Minimal Config (input stack)

```json
{
  input_filename: "synthetic_stack.tiff",
  thickness: 0.01,
  wavelengths: [650, 550, 450],
  C: [3e-9, 3e-9, 3e-9],
  S_i_hat: [1.0, 0.0, 0.0],
  output_filename: "recovered_stress.tiff"
}
```

## Command

```bash
image-to-stress params.json5
```

Optional CLI output override:

```bash
image-to-stress params.json5 --output recovered_stress.tiff
```

## Expected Outputs

- Stress map file (`[H, W, 3]`) when `output_filename` or `--output` is set
- Optional debug files when `debug: true`

## Commonly Used Controls

- `crop`: `[x1, x2, y1, y2]`
- `binning`: integer spatial binning factor
- Seeding controls:
  - `seeding.n_max`
  - `seeding.sigma_max`
- Correction for random stress orientation:
  - `correction` block with `enabled: true`, `order_param`, and `N` or `d`
- Optimisation controls:
  - `knot_spacing`, `spline_degree`
  - `max_iterations`, `tolerance`
  - `boundary_mask_file`, `boundary_values_files`, `boundary_weight`
  - `regularisation_weight` (or alias `regularization_weight`)
  - `external_potential_file` or `external_potential_gradient`

All keys are documented in [Configuration Reference](../configuration.md).

## Common Failure Modes

- `Missing stress-optic coefficient 'C'`:
  - add `C` or provide `calibration_file` with `C`
- `Missing sample thickness 'thickness'`:
  - set `thickness`
- `Either 'folderName' or 'input_filename' must be specified`:
  - set one input source
- `Use either external_potential_file or external_potential_gradient, not both`:
  - keep only one of the two keys
- `Boundary mask shape must be ...`:
  - ensure boundary files match image dimensions
- ``solver`is no longer supported`:
  - remove deprecated solver selection keys
