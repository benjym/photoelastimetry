# API Reference

This section contains auto-generated API docs from package source.

## Core Modules

- [main](main.md): CLI entry points and workflow orchestration
- [io](io.md): image/raw IO and channel operations
- [image](image.md): Stokes/Mueller operations and forward model helpers
- [calibrate](calibrate.md): calibration profile and fitting workflow
- [optimise](optimise.md): mean-stress recovery solver
- [seeding](seeding.md): phase-decomposed seeding and fringe resolution
- [bspline](bspline.md): B-spline stress/pressure field primitives
- [unwrapping](unwrapping.md): graph-cut angle unwrapping
- [visualisation](visualisation.md): terminal boundary-condition visualizations
- [plotting](plotting.md): plotting utilities

## Generator Modules

- [generate.disk](generate/disk.md)
- [generate.point_load](generate/point_load.md)
- [generate.strip_load](generate/strip_load.md)
- [generate.lithostatic](generate/lithostatic.md)
- [generate.inclined_plane](generate/inclined_plane.md)

## Intentionally Omitted

- `photoelastimetry.__init__`
- `photoelastimetry.generate.__init__`

These modules primarily provide package/export scaffolding and do not contain standalone runtime workflows.
