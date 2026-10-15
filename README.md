# photoelastimetry

[![Tests](https://github.com/benjym/photoelastimetry/actions/workflows/test.yml/badge.svg)](https://github.com/benjym/photoelastimetry/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/benjym/photoelastimetry/branch/main/graph/badge.svg)](https://codecov.io/gh/benjym/photoelastimetry)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Package for processing polarised images to measure stress in granular media.

## Install

```bash
pip install photoelastimetry
```

## CLI Tools

- `image-to-stress`
- `image-to-retardation` (quick first-branch optical inversion)
- `stress-to-image`
- `demosaic-raw`
- `calibrate-photoelastimetry`

Run `--help` on each command for usage.

For a quick retardation/direction map without stress fitting or fringe resolution:

```bash
image-to-retardation image.tiff --input-stokes 0 0 1
image-to-retardation image.tiff --bright-field bright.tiff --dark-field dark.tiff --format png
```

Defaults to linear illumination (`--input-stokes 1 0 0`) and float TIFF output
in radians. PNGs are fixed-scale previews, one per colour and quantity.
See [input layouts, branch assumptions and field correction](docs/user/workflows/image-to-retardation.md).

## Documentation

- Full docs: <https://benjym.github.io/photoelastimetry/>
- User docs (workflows/config): `docs/user/`
- Developer docs (setup/tests/architecture): `docs/developer/`
- API reference: `docs/reference/`

## Fastest Local Example

```bash
stress-to-image forward.json5
image-to-stress inverse.json5
```

For complete runnable examples, use `docs/user/quickstart.md` and workflow pages under `docs/user/workflows/`.

## Development

```bash
pip install -e ".[dev,docs]"
pytest
mkdocs build --strict
```

Contributor guidance is maintained in `docs/developer/`.
