# API Reference

This section provides detailed documentation for all modules, classes, and functions in the photoelastimetry package.

## Core Modules

- [disk](disk.md) - Elastic disk solution and photoelastic simulation
- [image](image.md) - Image processing and Mueller matrix operations
- [io](io.md) - Input/output operations for images and data
- [main](main.md) - Command-line interface entry points
- [plotting](plotting.md) - Visualization utilities and colormaps

## Optimiser Modules

The optimiser subpackage provides three complementary approaches for stress field recovery:

- [optimiser](optimiser.md) - Main optimiser module with high-level API
- [optimiser.stokes](stokes.md) - Stokes-based pixel-wise inversion
- [optimiser.intensity](intensity.md) - Intensity-based pixel-wise inversion
- [optimiser.equilibrium](equilibrium.md) - Global equilibrium-based inversion

## Quick Links

### Most Common Functions

**Stress Analysis:**
- `solver.recover_stress_map_stokes()` - Primary method for stress recovery
- `solver.compute_stokes_components()` - Compute Stokes parameters
- `solver.compute_normalized_stokes()` - Normalize Stokes components

**Image Processing:**
- `image.compute_retardance()` - Calculate optical retardance
- `image.compute_principal_angle()` - Calculate principal stress angle
- `image.mueller_matrix()` - Generate Mueller matrices

**Simulation:**
- `disk.simulate_four_step_polarimetry()` - Simulate photoelastic response

See individual module pages for complete documentation.
