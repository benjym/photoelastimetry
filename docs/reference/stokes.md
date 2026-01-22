# optimiser.stokes

Stokes-based pixel-wise stress inversion.

This module implements stress field recovery using normalised Stokes components computed from polarimetric measurements. This is the primary and recommended method for most photoelastic analysis tasks.

## Key Functions

- `compute_stokes_components()` - Calculate Stokes parameters from intensity measurements
- `compute_normalised_stokes()` - Normalise Stokes components for analysis
- `recover_stress_map_stokes()` - Main function for stress field recovery
- `predict_stokes()` - Forward model for validation

::: photoelastimetry.optimiser.stokes
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
