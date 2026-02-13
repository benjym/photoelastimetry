# Photoelastimetry Documentation

Photoelastimetry processes polarimetric images into stress fields and generates synthetic photoelastic images from stress fields.

Use this site by audience:

- **Users**: task-first guides for CLI workflows, JSON5 configs, and troubleshooting.
- **Developers**: setup, tests, architecture, contribution workflow, and docs/release maintenance.
- **Reference**: auto-generated API docs for package modules.

## Start Here

- New user: [Installation](user/installation.md) -> [Quickstart](user/quickstart.md)
- Existing user: [Workflow guides](user/index.md)
- Contributor: [Developer overview](developer/index.md)
- API lookup: [Reference index](reference/index.md)

## Command-Line Tools

- `image-to-stress`: invert photoelastic images to stress map
- `stress-to-image`: forward-simulate photoelastic images from stress map
- `demosaic-raw`: convert Bayer+polarisation raw files into channel stacks
- `calibrate-photoelastimetry`: fit calibration profile from known-load data

For exact config keys and aliases, use the canonical [Configuration Reference](user/configuration.md).
