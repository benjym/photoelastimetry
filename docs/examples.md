# Examples

This page provides practical examples for using the photoelastimetry package.

## Example 1: Elastic Disk Solution

You can generate a pre-set disk stress solution for validation using the parameters in `json/test.json5`:

```bash
python photoelastimetry/generate/disk.py
```

Then invert with:

```bash
image-to-stress json/test.json5
```

## Example 2: Direct Optimise Solver Usage

If you already have a seeded stress map, you can run the optimise solver directly:

```python
import numpy as np
from photoelastimetry.optimise import recover_mean_stress, stress_to_principal_invariants

# Seeded stress map shape [H, W, 3] ordered as [sigma_xx, sigma_yy, sigma_xy]
initial_stress_map = np.load("seeded_stress.npy")

delta_sigma, theta = stress_to_principal_invariants(initial_stress_map)

wrapper, coeffs = recover_mean_stress(
    delta_sigma,
    theta,
    knot_spacing=10,
    max_iterations=200,
    tolerance=1e-8,
    verbose=True,
)

sigma_xx, sigma_yy, sigma_xy = wrapper.get_stress_fields(coeffs)
stress_map = np.stack([sigma_xx, sigma_yy, sigma_xy], axis=-1)
```

## Example 3: Using Command Line Tools

### Process Raw Images

```bash
# First, demosaic raw polarimetric images
demosaic-raw raw_images/ --width 2448 --height 2048 --format png --all

# Create a parameter file (params.json5)
cat > params.json5 << EOF_JSON
{
  "folderName": "./raw_images",
  "C": 5e-11,
  "thickness": 0.005,
  "wavelengths": [450, 550, 650],
  "S_i_hat": [1.0, 0.0, 0.0],
  "crop": [200, 1800, 200, 2200],
  "knot_spacing": 12,
  "max_iterations": 150,
  "debug": false
}
EOF_JSON

# Run stress analysis
image-to-stress params.json5 --output stress_field.tiff
```

## Example 4: Forward Simulation

Generate synthetic photoelastic images from known stress fields:

```bash
cat > forward_params.json5 << EOF_JSON
{
  "stress_filename": "stress_field.npy",
  "thickness": 0.005,
  "wavelengths": [650, 550, 450],
  "C": [5e-11, 5e-11, 5e-11],
  "S_i_hat": [1.0, 0.0, 0.0],
  "scattering": 2.0,
  "output_filename": "synthetic_stack.tiff"
}
EOF_JSON

stress-to-image forward_params.json5
```

## Example 5: Calibrate on a Brazilian Disk Load Sweep

Use measured calibration frames to fit `C`, `S_i_hat`, and blank correction:

```bash
cat > calibration_params.json5 << EOF_JSON
{
  "method": "brazilian_disk",
  "wavelengths": [650, 550, 450],
  "thickness": 0.01,
  "geometry": {
    "radius_m": 0.01,
    "center_px": [256, 256],
    "pixels_per_meter": 20000,
    "edge_margin_fraction": 0.9,
    "contact_exclusion_fraction": 0.12
  },
  "dark_frame_file": "calibration/dark.npy",
  "blank_frame_file": "calibration/blank.npy",
  "load_steps": [
    {"load": 0.0, "image_file": "calibration/load_000.npy"},
    {"load": 150.0, "image_file": "calibration/load_150.npy"},
    {"load": 300.0, "image_file": "calibration/load_300.npy"},
    {"load": 450.0, "image_file": "calibration/load_450.npy"}
  ],
  "output_profile": "calibration/profile.json5",
  "output_report": "calibration/report.md",
  "output_diagnostics": "calibration/diagnostics.npz"
}
EOF_JSON

calibrate-photoelastimetry calibration_params.json5
```

Then use the profile during inversion:

```bash
image-to-stress params.json5 --output stress_field.tiff
```

Where `params.json5` includes:

```json
{
  "input_filename": "experiment/demosaiced.tiff",
  "thickness": 0.01,
  "calibration_file": "calibration/profile.json5",
  "debug": false
}
```

## Additional Resources

- See the [API Reference](reference/index.md) for detailed function documentation
- Check the [User Guide](user-guide.md) for parameter explanations
- Visit the [GitHub repository](https://github.com/benjym/photoelastimetry) for more examples
