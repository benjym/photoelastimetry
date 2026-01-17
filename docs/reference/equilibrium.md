# optimiser.equilibrium

Global stress measurement using Airy stress function.

This module implements a global inversion approach that solves for an Airy stress function across the entire domain simultaneously. It ensures mechanical equilibrium by construction and enforces smoothness globally.

## Key Functions

- `recover_stress_global()` - Main function for global stress recovery
- `loss_and_gradient()` - Optimization loss function function

::: photoelastimetry.optimiser.equilibrium
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
