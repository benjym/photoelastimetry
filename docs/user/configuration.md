# Configuration Reference

This is the canonical reference for CLI JSON5 parameters.

## Conventions

- Wavelengths accept meters or nanometers:
  - values `> 1e-6` are interpreted as nanometers and converted to meters
- `S_i_hat` can have length 2 or 3:
  - `[S1_hat, S2_hat]` or `[S1_hat, S2_hat, S3_hat]`
- Stress map default order is `[sigma_xx, sigma_yy, sigma_xy]`

## Parameter Precedence

### image-to-stress

- Input source: either `folderName` or `input_filename` is required
- If `calibration_file` is set:
  - missing `C`, `S_i_hat`, `wavelengths` are filled from the profile
  - blank correction is applied
- For output path precedence:
  - if `output_filename` exists in JSON, it takes precedence over CLI `--output`

### stress-to-image

- Optional `p_filename` loads fallback parameters from a second JSON5 file
- Precedence order:
  1. explicit keys in current params
  2. fallback keys from `p_filename`
  3. calibration profile values (`C`, `S_i_hat`, `wavelengths`) if `calibration_file` is set

## image-to-stress Keys

Required (directly or via calibration):

- `thickness`
- `wavelengths`
- `C`
- `S_i_hat`
- plus one input source: `folderName` or `input_filename`

Common optional keys:

- `output_filename`
- `crop`: `[x1, x2, y1, y2]`
- `binning`
- `debug`

Seeding keys:

- `seeding.n_max`
- `seeding.sigma_max`

Disorder Correction keys:

Optional block `correction`:

- `correction.enabled`: boolean (default `false`)
- `correction.order_param`: float, [0, 1] (order parameter |m|)
- `correction.N`: float (number of grain encounters)
- `correction.d`: float (particle diameter, used to estimate N if N is missing)

Optimisation keys:

- `knot_spacing`
- `spline_degree`
- `boundary_mask_file`
- `boundary_values_files`
- `boundary_weight`
- `regularisation_weight` (alias `regularization_weight`)
- `regularisation_order`
- `external_potential_file`
- `external_potential_gradient` (`[dVdx, dVdy]`)
- `max_iterations`
- `tolerance`
- `verbose`
- `debug`

Not supported:

- `solver`
- `global_solver`
- `global_mean_stress`

## stress-to-image Keys

Required:

- `stress_filename` (legacy alias: `s_filename`)
- `thickness` (legacy alias: `t`)
- `wavelengths` (legacy alias: `lambda_light`)
- `C`

Optional:

- `S_i_hat`
- `calibration_file`
- `stress_order`: `xx_yy_xy` (default) or `xy_yy_xx` (legacy)
- `scattering`
- `output_filename`
- `p_filename` (fallback param file)

## demosaic-raw CLI Flags

- positional: `input_file`
- `--width` (default `4096`)
- `--height` (default `3000`)
- `--dtype` (`uint8` or `uint16`)
- `--output-prefix`
- `--format` (`tiff` or `png`)
- `--all` (recursive processing)
- `--mode` (`auto`, `single`, `average`, `series`)
- `--average-method` (`mean` or `median`)
- `--start` (frame start index, inclusive)
- `--stop` (frame stop index, exclusive)
- `--step` (frame stride, default `1`)

Notes:

- `--mode auto` chooses:
  - `single` for a `.raw` input file
  - `average` for a recording directory
- Recording directory mode expects `0000000/frame*.raw`.
- Metadata is auto-loaded from `recordingMetadata.json` when available.

## calibrate-photoelastimetry Keys

Required:

- `method`: `brazilian_disk` or `coupon_test`
- `wavelengths`
- `thickness`
- `geometry` (method-specific)
- `load_steps`

Method-specific required geometry:

- `brazilian_disk`: `radius_m`, `center_px`, `pixels_per_meter`
- `coupon_test`: `gauge_roi_px`, `coupon_width_m`

Common optional keys:

- `dark_frame_file` + `blank_frame_file` (both or neither)
- `fit.max_points`, `fit.seed`, `fit.loss`, `fit.f_scale`, `fit.max_nfev`
- `fit.initial_C`, `fit.initial_S_i_hat`
- `fit.s3_identifiability_threshold`, `fit.prior_weight`, `fit.c_relative_bounds`
- `output_profile`, `output_report`, `output_diagnostics`

Load step input notes:

- Each `load_steps[i].image_file` may be either:
  - a demosaiced stack (`[H, W, n_wavelengths, 4]`), or
  - a raw frame (`.raw`) with nearby `recordingMetadata.json`
- Raw calibration inputs are demosaiced internally to `[H, W, 3, 4]` using channels `R, G1, B`.

## calibrate-photoelastimetry CLI Flags

- positional: `json_filename`
- `--interactive` (launch click-based geometry wizard)
- `--save-config <path>` (write updated config after interactive geometry selection)
