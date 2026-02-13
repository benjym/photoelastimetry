# Workflow: demosaic-raw

Convert raw polarimetric Bayer files into channel-separated outputs.

## Prerequisites

- Raw `.raw` file(s)
- Either:
  - explicit `--width`/`--height` and optional `--dtype`, or
  - `recordingMetadata.json` in your capture folder for pixel-format auto detection when using higher-level loaders

## Single File Command

```bash
demosaic-raw frame.raw --width 2448 --height 2048 --dtype uint16 --format tiff
```

## Batch Command

```bash
demosaic-raw captures/ --format png --all
```

## Output Formats

- `--format tiff`:
  - writes `<prefix>_demosaiced.tiff`
- `--format png`:
  - writes `<prefix>_0deg.png`, `<prefix>_45deg.png`, `<prefix>_90deg.png`, `<prefix>_135deg.png`

## Common Failure Modes

- `When using --all flag, input_file must be a directory`:
  - pass a directory path with `--all`
- File size mismatch errors:
  - verify `--width`, `--height`, and `--dtype` match capture settings
- Unsupported `pixelFormat`:
  - use one of the supported Bayer 8/10/12 PFNC codes documented by the CLI and IO module
