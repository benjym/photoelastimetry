import json
from pathlib import Path

import numpy as np
import pytest

from photoelastimetry import calibrate, io
from photoelastimetry.generate.disk import diametrical_stress_cartesian
from photoelastimetry.image import simulate_four_step_polarimetry


def _make_synthetic_calibration_case(tmp_path, noisy=False, inconsistent_shapes=False):
    height, width = 34, 34
    thickness = 0.01
    wavelengths = np.array([650e-9, 550e-9, 450e-9], dtype=float)
    c_true = np.array([2.2e-9, 2.5e-9, 2.8e-9], dtype=float)
    s_true = np.array([0.82, 0.18, 0.0], dtype=float)

    radius_m = 0.008
    pixels_per_meter = 1800.0
    cx = width / 2.0
    cy = height / 2.0

    x = (np.arange(width) - cx) / pixels_per_meter
    y = (np.arange(height) - cy) / pixels_per_meter
    X, Y = np.meshgrid(x, y)
    disk_mask = X**2 + Y**2 <= radius_m**2

    loads = [0.0, 2.0, 4.0, 6.0, 8.0]

    # Per-channel/per-polariser detector model: corrected = (raw - offset) * scale
    offset = np.array(
        [
            [0.03, 0.032, 0.031, 0.029],
            [0.028, 0.030, 0.029, 0.027],
            [0.034, 0.036, 0.033, 0.032],
        ],
        dtype=float,
    )
    scale = np.array(
        [
            [0.95, 0.98, 0.97, 0.96],
            [0.92, 0.94, 0.93, 0.95],
            [0.90, 0.91, 0.89, 0.92],
        ],
        dtype=float,
    )

    dark = np.broadcast_to(offset[np.newaxis, np.newaxis, :, :], (height, width, 3, 4)).copy()
    blank = np.broadcast_to(
        (offset + 1.0 / scale)[np.newaxis, np.newaxis, :, :], (height, width, 3, 4)
    ).copy()

    if noisy:
        rng = np.random.default_rng(42)
        dark += rng.normal(scale=1e-6, size=dark.shape)
        blank += rng.normal(scale=1e-6, size=blank.shape)

    dark_file = tmp_path / "dark.npy"
    blank_file = tmp_path / "blank.npy"
    io.save_image(str(dark_file), dark.astype(np.float32))
    io.save_image(str(blank_file), blank.astype(np.float32))

    load_steps = []
    rng = np.random.default_rng(123)

    for idx, load in enumerate(loads):
        sigma_xx, sigma_yy, sigma_xy = diametrical_stress_cartesian(X, Y, P=load, R=radius_m)

        corrected = np.zeros((height, width, 3, 4), dtype=float)
        for c, wl in enumerate(wavelengths):
            i0, i45, i90, i135 = simulate_four_step_polarimetry(
                sigma_xx,
                sigma_yy,
                sigma_xy,
                c_true[c],
                1.0,
                thickness,
                wl,
                s_true,
            )
            corrected[:, :, c, 0] = i0
            corrected[:, :, c, 1] = i45
            corrected[:, :, c, 2] = i90
            corrected[:, :, c, 3] = i135

        # Outside disk behaves like unstressed background.
        corrected[~disk_mask, :, :] = 1.0

        raw = corrected / scale[np.newaxis, np.newaxis, :, :] + offset[np.newaxis, np.newaxis, :, :]
        if noisy:
            raw += rng.normal(scale=5e-5, size=raw.shape)

        if inconsistent_shapes and idx == len(loads) - 1:
            raw = raw[:-1, :, :, :]

        image_file = tmp_path / f"load_{idx}.npy"
        io.save_image(str(image_file), raw.astype(np.float32))
        load_steps.append({"load": float(load), "image_file": str(image_file)})

    config = {
        "method": "brazilian_disk",
        "wavelengths": wavelengths.tolist(),
        "thickness": thickness,
        "nu": 1.0,
        "geometry": {
            "radius_m": radius_m,
            "center_px": [cx, cy],
            "pixels_per_meter": pixels_per_meter,
            "edge_margin_fraction": 0.8,
            "contact_exclusion_fraction": 0.35,
        },
        "load_steps": load_steps,
        "dark_frame_file": str(dark_file),
        "blank_frame_file": str(blank_file),
        "fit": {
            "max_points": 1500,
            "seed": 11,
            "loss": "soft_l1",
            "f_scale": 0.05,
            "max_nfev": 250,
            "prior_weight": 10000.0 if noisy else 0.0,
            "c_relative_bounds": [0.5, 2.0] if noisy else None,
            "initial_C": [2.0e-9, 2.0e-9, 2.0e-9],
            "initial_S_i_hat": [0.8, 0.1, 0.0],
        },
        "output_profile": str(tmp_path / "calibration_profile.json5"),
        "output_report": str(tmp_path / "calibration_report.md"),
        "output_diagnostics": str(tmp_path / "calibration_diagnostics.npz"),
    }

    return config, c_true, s_true


def test_calibration_residuals_near_zero_for_noiseless_data(tmp_path):
    config, c_true, s_true = _make_synthetic_calibration_case(tmp_path, noisy=False)
    validated = calibrate.validate_calibration_config(config)
    dataset = calibrate._build_dataset(validated)

    params = np.concatenate([c_true, s_true], axis=0)
    residual = calibrate.calibration_residuals(params, dataset)

    assert residual.ndim == 1
    assert np.sqrt(np.mean(residual**2)) < 1e-6


def test_fit_recovers_c_and_s_i_hat_on_noisy_data(tmp_path):
    config, c_true, s_true = _make_synthetic_calibration_case(tmp_path, noisy=True)
    result = calibrate.run_calibration(config)
    profile = result["profile"]

    c_fit = np.asarray(profile["C"], dtype=float)
    s_fit = np.asarray(profile["S_i_hat"], dtype=float)

    assert np.allclose(c_fit, c_true, rtol=0.2, atol=0.0)
    assert np.allclose(s_fit[:2], s_true[:2], atol=0.12)
    assert profile["fit_metrics"]["success"]


def test_blank_correction_coefficients_and_application():
    rng = np.random.default_rng(4)
    corrected_true = rng.uniform(0.0, 1.0, size=(7, 9, 3, 4))

    offset = np.array(
        [
            [0.01, 0.02, 0.03, 0.04],
            [0.02, 0.03, 0.04, 0.05],
            [0.03, 0.04, 0.05, 0.06],
        ],
        dtype=float,
    )
    scale = np.array(
        [
            [0.9, 0.85, 0.88, 0.91],
            [0.95, 0.93, 0.94, 0.92],
            [0.87, 0.89, 0.86, 0.90],
        ],
        dtype=float,
    )

    raw = corrected_true / scale[np.newaxis, np.newaxis, :, :] + offset[np.newaxis, np.newaxis, :, :]
    dark = np.broadcast_to(offset[np.newaxis, np.newaxis, :, :], raw.shape)
    blank = np.broadcast_to((offset + 1.0 / scale)[np.newaxis, np.newaxis, :, :], raw.shape)

    correction = calibrate.compute_blank_correction(dark, blank)
    corrected = calibrate.apply_blank_correction(raw, correction)

    assert np.allclose(corrected, corrected_true, rtol=1e-7, atol=1e-7)


def test_validation_errors_for_bad_configuration(tmp_path):
    config, _, _ = _make_synthetic_calibration_case(tmp_path, noisy=False)

    no_noload = dict(config)
    no_noload["load_steps"] = [dict(step, load=abs(step["load"]) + 1.0) for step in config["load_steps"]]
    with pytest.raises(ValueError, match="no-load"):
        calibrate.validate_calibration_config(no_noload)

    bad_geometry = dict(config)
    bad_geometry["geometry"] = dict(config["geometry"])
    bad_geometry["geometry"]["radius_m"] = -0.1
    with pytest.raises(ValueError, match="radius_m"):
        calibrate.validate_calibration_config(bad_geometry)

    short_steps = dict(config)
    short_steps["load_steps"] = config["load_steps"][:3]
    with pytest.raises(ValueError, match="at least 4"):
        calibrate.validate_calibration_config(short_steps)


def test_validation_error_for_inconsistent_image_shapes(tmp_path):
    config, _, _ = _make_synthetic_calibration_case(tmp_path, noisy=False, inconsistent_shapes=True)
    validated = calibrate.validate_calibration_config(config)

    with pytest.raises(ValueError, match="share shape"):
        calibrate._build_dataset(validated)


def test_end_to_end_writes_profile_report_and_diagnostics(tmp_path):
    config, _, _ = _make_synthetic_calibration_case(tmp_path, noisy=False)
    result = calibrate.run_calibration(config)

    assert Path(result["profile_file"]).exists()
    assert Path(result["report_file"]).exists()
    assert Path(result["diagnostics_file"]).exists()

    with open(result["profile_file"], "r") as f:
        profile = json.load(f)

    required_keys = {
        "version",
        "method",
        "wavelengths",
        "C",
        "S_i_hat",
        "blank_correction",
        "fit_metrics",
        "provenance",
    }
    assert required_keys.issubset(profile.keys())

    loaded = calibrate.load_calibration_profile(result["profile_file"])
    assert loaded["method"] == "brazilian_disk"
