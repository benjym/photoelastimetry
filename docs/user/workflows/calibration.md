# Workflow: calibrate-photoelastimetry

Fit calibration profile values (`C`, `S_i_hat`, blank correction) from known-load image data.

## Prerequisites

- Calibration input for each load step can be either:
  - a stack with shape `[H, W, n_wavelengths, 4]` (`.npy/.tiff/.tif`), or
  - a single raw frame (`.raw`) plus nearby `recordingMetadata.json`
- For raw-frame input:
  - metadata is auto-loaded from `recordingMetadata.json`
  - frame is demosaiced internally to `[H, W, 3, 4]` (RGB = `R, G1, B`)
- At least 4 load steps total:
  - at least 1 no-load step (`load ~= 0`)
  - at least 3 non-zero load steps
- Method-specific geometry:
  - `brazilian_disk`: `radius_m`, `center_px`, `pixels_per_meter`
  - `coupon_test`: `gauge_roi_px`, `coupon_width_m`

## Example Config (Brazilian disk)

```json
{
  method: "brazilian_disk",
  wavelengths: [650, 550, 450],
  thickness: 0.01,
  geometry: {
    radius_m: 0.01,
    center_px: [256, 256],
    pixels_per_meter: 20000,
    edge_margin_fraction: 0.9,
    contact_exclusion_fraction: 0.12
  },
  load_steps: [
    {load: 0.0, image_file: "calib/load_000.npy"},
    {load: 150.0, image_file: "calib/load_150.npy"},
    {load: 300.0, image_file: "calib/load_300.npy"},
    {load: 450.0, image_file: "calib/load_450.npy"}
  ],
  dark_frame_file: "calib/dark.npy",
  blank_frame_file: "calib/blank.npy",
  output_profile: "calib/profile.json5",
  output_report: "calib/report.md",
  output_diagnostics: "calib/diagnostics.npz"
}
```

## Command

```bash
calibrate-photoelastimetry calibration.json5
```

Interactive geometry setup:

```bash
calibrate-photoelastimetry calibration.json5 --interactive --save-config calib/calibration_updated.json
```

In interactive mode:
- `brazilian_disk`:
  - left-click multiple points on the disk circumference (at least 3)
  - a fitted circle overlay updates live as you add points
  - ROI overlay and `roi_pixels` count update live
  - right-click to undo the last point
  - click `Done` when the overlay circle matches (or `Reset` to start over)
  - `Done` is blocked if ROI would be empty for the current circle
- `coupon_test`: click top-left, then bottom-right corners of the gauge ROI

## Expected Outputs

- Calibration profile JSON5 (`output_profile`)
- Markdown report (`output_report`)
- Diagnostics archive (`output_diagnostics`)
- Diagnostics figure PNG (`<output_report stem>_fit.png`)

## Use the Profile in Inversion

```json
{
  input_filename: "experiment_stack.tiff",
  thickness: 0.01,
  calibration_file: "calib/profile.json5",
  output_filename: "recovered_stress.tiff"
}
```

`image-to-stress` fills missing `C`, `S_i_hat`, and `wavelengths` from the profile and applies blank correction automatically.

## Common Failure Modes

- `Calibration requires at least 4 load_steps`:
  - add enough load steps
- `at least one no-load` / `at least three non-zero` errors:
  - rebalance load list
- `Provide both dark_frame_file and blank_frame_file, or neither`:
  - specify both or remove both
- Shape mismatch across calibration images:
  - make all load-step images exactly the same post-load shape
  - for raw inputs, ensure all captures share dimensions and pixel format
- Raw load step fails to decode:
  - verify `recordingMetadata.json` exists near each raw frame and includes valid pixel metadata

## Beginner Checklist

1. Start with one no-load frame and at least three non-zero load frames.
2. Verify all calibration inputs are the same capture geometry and dimensions.
3. Run once with `--interactive` to set geometry visually.
4. Check the diagnostics plot (`*_fit.png`):
   - measured vs synthetic maps should look qualitatively similar
   - residual maps should be small and structure-free in the ROI
5. If residuals are structured:
   - revisit geometry clicks
   - verify load magnitudes and thickness units
   - include dark/blank correction frames if available
