# optimiser.intensity

Intensity-based pixel-wise stress inversion.

This module provides an alternative approach that works directly with raw polarization intensities rather than Stokes components. This can be beneficial when raw intensities provide more reliable measurements.

## Key Functions

- `predict_intensity()` - Forward model for intensity prediction
- `recover_stress_map_intensity()` - Main function for intensity-based stress recovery
- `compare_stokes_vs_intensity()` - Compare Stokes and intensity-based methods

::: photoelastimetry.optimiser.intensity
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
