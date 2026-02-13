# Python API Workflow

Use Python when you want scripted pipelines or notebook workflows.

## Invert Images to Stress

```python
import json5
from photoelastimetry.main import image_to_stress

with open("params.json5", "r") as f:
    params = json5.load(f)

stress = image_to_stress(params)
print(stress.shape)  # [H, W, 3]
```

## Forward-Simulate Images from Stress

```python
import json5
from photoelastimetry.main import stress_to_image

with open("forward.json5", "r") as f:
    params = json5.load(f)

synthetic = stress_to_image(params)
print(synthetic.shape)  # [H, W, n_wavelengths, 4]
```

## Run Calibration in Python

```python
import json5
from photoelastimetry.calibrate import run_calibration

with open("calibration.json5", "r") as f:
    config = json5.load(f)

result = run_calibration(config)
print(result["profile_file"])
```

## Direct Mean-Stress Recovery (advanced)

```python
import numpy as np
from photoelastimetry.optimise import stress_to_principal_invariants, recover_mean_stress

initial_stress = np.load("seeded_stress.npy")  # [H, W, 3]
delta_sigma, theta = stress_to_principal_invariants(initial_stress)

wrapper, coeffs = recover_mean_stress(
    delta_sigma,
    theta,
    knot_spacing=8,
    max_iterations=200,
    tolerance=1e-8,
    verbose=True,
)

sigma_xx, sigma_yy, sigma_xy = wrapper.get_stress_fields(coeffs)
```

For parameter details and key aliases, use the canonical [Configuration Reference](configuration.md).
