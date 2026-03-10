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
from photoelastimetry.optimise import recover_mean_stress
from photoelastimetry.seeding import phase_decomposed_seeding

seed = phase_decomposed_seeding(
    image_stack,
    wavelengths,
    c_values,
    nu=1.0,
    L=0.01,
    S_i_hat=[1.0, 0.0, 0.0],
)

wrapper, coeffs = recover_mean_stress(
    seed.delta_sigma,
    seed.theta,
    knot_spacing=8,
    max_iterations=200,
    tolerance=1e-8,
    verbose=True,
    initial_stress_map=seed.to_stress_map(K=0.5),
)

sigma_xx, sigma_yy, sigma_xy = wrapper.get_stress_fields(coeffs)
```

For parameter details and key aliases, use the canonical [Configuration Reference](configuration.md).
