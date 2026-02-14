# Workflow: demosaic-raw

Convert raw polarimetric Bayer files into channel-separated outputs.

## Prerequisites

- Input can be:
  - one raw frame file (for single-frame processing), or
  - one recording directory containing `0000000/frame*.raw`
- Metadata can come from:
  - `recordingMetadata.json` (auto-loaded when available), or
  - explicit `--width`/`--height` and optional `--dtype`

## Single File Command

```bash
demosaic-raw frame.raw --width 2448 --height 2048 --dtype uint16 --format tiff
```

## Recording Directory Commands

```bash
# auto mode for a directory defaults to averaging
demosaic-raw capture_dir/ --format png

# explicit average mode
demosaic-raw capture_dir/ --mode average --average-method median --format tiff

# explicit time-series mode
demosaic-raw capture_dir/ --mode series --format png
```

## Frame Range Selection

Use `--start` (inclusive), `--stop` (exclusive), and `--step`:

```bash
demosaic-raw capture_dir/ --mode average --start 10 --stop 50 --step 2
demosaic-raw capture_dir/ --mode series --start 0 --stop 20
```

For recursive `--all`, range selection is also applied to the discovered `.raw` list.

## Output Formats

- `--format tiff`:
  - writes `<prefix>_demosaiced.tiff`
- `--format png`:
  - writes `<prefix>_0deg.png`, `<prefix>_45deg.png`, `<prefix>_90deg.png`, `<prefix>_135deg.png`

## Output Location Defaults

- Single file input:
  - output prefix defaults to input filename stem
- Recording directory with `--mode average` (or `--mode auto`):
  - outputs under `<recording_dir>/average/`
- Recording directory with `--mode series`:
  - outputs under `<recording_dir>/series/`

## Common Failure Modes

- `When using --all flag, input_file must be a directory`:
  - pass a directory path with `--all`
- `Mode 'single' requires a .raw file input, not a directory`:
  - use `--mode average` or `--mode series` for recording directories
- `Mode 'average' requires a directory input`:
  - provide a recording directory for averaging
- `No .raw files selected after applying frame range...`:
  - adjust `--start/--stop/--step` so at least one frame is selected
- File size mismatch errors:
  - verify metadata and/or `--width`, `--height`, `--dtype` match capture settings
- Unsupported `pixelFormat`:
  - use one of the supported Bayer 8/10/12 PFNC codes documented by the CLI and IO module
