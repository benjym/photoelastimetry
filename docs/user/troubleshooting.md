# Troubleshooting

## image-to-stress

### Error: missing `C`, `thickness`, `wavelengths`, or `S_i_hat`

- Provide keys directly in JSON5, or provide `calibration_file` that contains them.

### Error: boundary file shape mismatch

- Boundary mask/value files must match the loaded image size exactly (`H x W`).

### Error: `Use either external_potential_file or external_potential_gradient, not both`

- Keep only one potential input mechanism.

### Error: deprecated solver configuration keys

- Remove `solver`, `global_solver`, and `global_mean_stress` keys.

## stress-to-image

### Error: missing stress map path

- Use `stress_filename` (or legacy `s_filename`).

### Error: `C` length mismatch

- `C` must be scalar or same length as wavelength count.

### Error: unsupported output extension

- Use `.tiff/.tif/.npy/.raw` for stacks or `.png/.jpg/.jpeg` for plots.

## demosaic-raw

### File size mismatch

- Verify width, height, dtype, and raw packing assumptions.

### Unsupported `pixelFormat`

- Use supported Bayer PFNC codes, or provide `--dtype` explicitly for manual decoding.

## calibration

### Error: not enough load steps

- Need at least 4 total, with at least one near-zero load and three non-zero loads.

### Error: image stack shape mismatch

- Every calibration frame (all load steps, dark, blank) must have identical shape `[H, W, n_wavelengths, 4]`.

### Error: incomplete dark/blank settings

- Provide both `dark_frame_file` and `blank_frame_file`, or neither.

## General Debug Checklist

- Confirm all paths are relative to your current working directory (or use absolute paths).
- Verify data shape conventions before running the CLI.
- Run commands with `--help` to confirm flags and expected input format.
