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
h, w = 96, 96
y, x = np.indices((h, w), dtype=float)
s_xx = 2.0e6 + 1.0e5 * (x / w)
s_yy = 1.5e6 + 8.0e4 * (y / h)
s_xy = 2.5e4 * np.sin(2*np.pi*x/w) * np.sin(2*np.pi*y/h)
stress = np.stack([s_xx, s_yy, s_xy], axis=-1)
np.save("stress.npy", stress)
PY
```

## 3. Generate Synthetic Polarimetric Images

Create `forward.json5`:

```json
{
  stress_filename: "stress.npy",
  thickness: 0.01,
  wavelengths: [650, 550, 450],
  C: [3e-9, 3e-9, 3e-9],
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

## 4. Invert Images Back to Stress

Create `inverse.json5`:

```json
{
  input_filename: "synthetic_stack.tiff",
  thickness: 0.01,
  wavelengths: [650, 550, 450],
  C: [3e-9, 3e-9, 3e-9],
  S_i_hat: [1.0, 0.0, 0.0],
  knot_spacing: 8,
  max_iterations: 120,
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

## Next Steps

- Use the full [image-to-stress workflow](workflows/image-to-stress.md)
- Learn all accepted keys in [Configuration Reference](configuration.md)
