# Examples

This page provides practical examples for using the photoelastimetry package.

## Example 1: Elastic Disk Solution

You can generate a pre-set disk stress solution for validation using the parameters in `json/test.json5`:

```bash
python photoelastimetry/generate/disk.py
```

This can be inverted to recover the stress field using the standard solvers via

```bash
image-to-stress json/test.json5
```

## Example 2: Basic Stress Analysis

Analyze a set of photoelastic images to extract stress fields:

```python
import photoelastimetry.optimiser as optimiser
import numpy as np

# Load your polarimetric images (4 angles: 0°, 45°, 90°, 135°)
I0 = np.load('image_0deg.npy')
I45 = np.load('image_45deg.npy')
I90 = np.load('image_90deg.npy')
I135 = np.load('image_135deg.npy')

# Stack intensities
intensities = np.stack([I0, I45, I90, I135], axis=-1)

# Compute Stokes components
S = solver.compute_stokes_components(intensities)

# Normalize Stokes components
S_normalized = solver.compute_normalized_stokes(S)

# Material properties
C = 5e-11  # Stress-optic coefficient (1/Pa)
t = 0.005  # Sample thickness (m)
wavelength = 550e-9  # Wavelength (m)
nu = 1.0  # Solid fraction

# Recover stress field
stress_map = solver.recover_stress_map_stokes(
    S_normalized, C, nu, t, wavelength
)

# Extract stress components
sigma_xx = stress_map[..., 0]
sigma_yy = stress_map[..., 1]
sigma_xy = stress_map[..., 2]
```

## Example 3: Using Command Line Tools

### Process Raw Images

```bash
# First, demosaic raw polarimetric images
demosaic-raw raw_images/ --width 2448 --height 2048 --format png --all

# Create a parameter file (params.json5)
cat > params.json5 << EOF
{
  "folderName": "./raw_images",
  "C": 5e-11,
  "thickness": 0.005,
  "wavelengths": [450, 550, 650],
  "S_i_hat": [1.0, 0.0, 0.0],
  "crop": [200, 1800, 200, 2200],
  "debug": false
}
EOF

# Run stress analysis
image-to-stress params.json5 --output stress_field.png
```

## Example 4: Comparing Solver Methods

Compare results from different stress inversion methods:

```python
import photoelastimetry.optimiser.intensity as intensity_optimiser
import photoelastimetry.optimiser.stokes as stokes_optimiser

# Using intensity-based method
stress_intensity = intensity_solver.recover_stress_map_intensity(
    intensities, C, nu, t, wavelength
)

# Using Stokes-based method
S = stokes_solver.compute_stokes_components(intensities)
S_norm = stokes_solver.compute_normalized_stokes(S)
stress_stokes = stokes_solver.recover_stress_map_stokes(
    S_norm, C, nu, t, wavelength
)

# Compare methods
comparison = intensity_solver.compare_stokes_vs_intensity(
    intensities, C, nu, t, wavelength
)
```

## Example 5: Global Equilibrium Solver

Use the equilibrium-based solver for mechanical consistency:

```python
import photoelastimetry.optimiser.equilibrium as eq_optimiser

# First get local solution
stress_local = solver.recover_stress_map_stokes(
    S_normalized, C, nu, t, wavelength
)

# Grid spacing
dx = 1.0  # meters
dy = 1.0  # meters

# Refine using equilibrium constraints
stress_global = eq_solver.recover_stress_field_global(
    stress_local, dx, dy, max_iterations=1000
)

# Compare local vs global solutions
comparison = eq_solver.compare_local_vs_global(
    stress_local, stress_global, dx, dy
)
```

## Example 6: Forward Simulation

Generate synthetic photoelastic images from known stress fields:

```bash
# Create parameter file for forward simulation
cat > forward_params.json5 << EOF
{
  "p_filename": "experimental_params.json5",
  "stress_filename": "stress_field.npy",
  "t": 0.005,
  "lambda_light": 550e-9,
  "C": 5e-11,
  "scattering": 2.0,
  "output_filename": "synthetic_image.png"
}
EOF

# Run forward simulation
stress-to-image forward_params.json5
```

## Additional Resources

- See the [API Reference](reference/index.md) for detailed function documentation
- Check the [User Guide](user-guide.md) for parameter explanations
- Visit the [GitHub repository](https://github.com/benjym/photoelastimetry) for more examples
