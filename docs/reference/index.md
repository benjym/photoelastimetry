# API Reference

This section provides detailed documentation for all modules, classes, and functions in the photoelastimetry package.

## Core Modules

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

## Generation Modules

- [generate.disk](generate/disk.md) - Generate synthetic disk data
- [generate.point_load](generate/point_load.md) - Generate point load stress fields
- [generate.lithostatic](generate/lithostatic.md) - Generate lithostatic stress fields
