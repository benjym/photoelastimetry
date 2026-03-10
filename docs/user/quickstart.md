# Quickstart

This quickstart runs a complete forward+inverse cycle using local synthetic data.

## Prerequisites

- Installed package (`pip install photoelastimetry`)
- Shell with `python` and CLI scripts on `PATH`

## 1. Create a Working Directory

```bash
mkdir -p quickstart
cd quickstart
```

## 2. Create a Synthetic Stress Field

```bash
python - <<'PY'
import numpy as np
import photoelastimetry.io
h, w = 96, 96
y, x = np.indices((h, w), dtype=float)
s_xx = 2.0e6 + 1.0e5 * (x / w)
s_yy = 1.5e6 + 8.0e4 * (y / h)
s_xy = 2.5e4 * np.sin(2*np.pi*x/w) * np.sin(2*np.pi*y/h)
stress = np.stack([s_xx, s_yy, s_xy], axis=-1)
photoelastimetry.io.save_image("stress.tiff", stress)
PY
```

You should now have `stress.tiff` in your working directory, which is a synthetic stress field of shape `[96, 96, 3]` (s_xx, s_yy, s_xy). You can visualize it with your preferred image viewer (we recommend [ImageJ](https://imagej.nih.gov/ij/)) or using `photoelastimetry.io.load_image` in Python.

## 3. Generate Synthetic Polarimetric Images

Create `forward.json5`:

```json
{
  stress_filename: "stress.tiff",
  thickness: 0.01,
  wavelengths: [650, 550, 450],
  C: [3e-10, 3e-10, 3e-10],
  S_i_hat: [1.0, 0.0, 0.0],
  output_filename: "synthetic_stack.tiff"
}
```

Run:

```bash
stress-to-image forward.json5
```

Expected output:

- `synthetic_stack.tiff` (`[H, W, 3, 4]`)

Note: many TIFF viewers display this as a stack with `4` frames and `3` channels (on-disk order `TCYX`), which can look like tiled panels. This is expected.

## 4. Invert Images Back to Stress

Create boundary-condition files to anchor absolute pressure:

```bash
python - <<'PY'
import numpy as np
import photoelastimetry.io

stress, _ = photoelastimetry.io.load_image("stress.tiff")
h, w, _ = stress.shape

boundary_mask = np.zeros((h, w), dtype=np.uint8)
boundary_mask[0, :] = 1
boundary_mask[-1, :] = 1
boundary_mask[:, 0] = 1
boundary_mask[:, -1] = 1

b_xx = np.full((h, w), np.nan, dtype=float)
b_yy = np.full((h, w), np.nan, dtype=float)
b_xy = np.full((h, w), np.nan, dtype=float)

mask = boundary_mask.astype(bool)
b_xx[mask] = stress[..., 0][mask]
b_yy[mask] = stress[..., 1][mask]
b_xy[mask] = stress[..., 2][mask]

np.save("boundary_mask.npy", boundary_mask)
np.save("boundary_xx.npy", b_xx)
np.save("boundary_yy.npy", b_yy)
np.save("boundary_xy.npy", b_xy)
PY
```

Create `inverse.json5`:

```json
{
  input_filename: "synthetic_stack.tiff",
  thickness: 0.01,
  wavelengths: [650, 550, 450],
  C: [3e-10, 3e-10, 3e-10],
  S_i_hat: [1.0, 0.0, 0.0],
  knot_spacing: 8,
  boundary_mask_file: "boundary_mask.npy",
  boundary_values_files: {
    xx: "boundary_xx.npy",
    yy: "boundary_yy.npy",
    xy: "boundary_xy.npy"
  },
  debug: false,
  output_filename: "recovered_stress.tiff"
}
```

Run:

```bash
image-to-stress inverse.json5
```

Expected output:

- `recovered_stress.tiff` (`[H, W, 3]`)

Without boundary conditions, `image-to-stress` only recovers stress up to an additive hydrostatic offset (zero-mean gauge), so absolute `s_xx`/`s_yy` can look wrong even if fringe structure is matched.

## Next Steps

- Use the full [image-to-stress workflow](workflows/image-to-stress.md)
- Learn all accepted keys in [Configuration Reference](configuration.md)
