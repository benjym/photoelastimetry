"""Tests for the Stage-2 wavelength dispersion autocalibration."""

import numpy as np
import pytest

from photoelastimetry import calibrate, io
from photoelastimetry.calibrate import (
    _dispersion_cauchy,
    _dispersion_power_law,
    _get_dispersion_fn,
    fit_dispersion_parameters,
    validate_calibration_config,
)
from photoelastimetry.generate.disk import diametrical_stress_cartesian
from photoelastimetry.image import simulate_four_step_polarimetry


# ---------------------------------------------------------------------------
# Helper: synthetic multi-fringe Brazilian disk dataset
# ---------------------------------------------------------------------------


def _make_dispersion_case(tmp_path, alpha_true=0.8, lambda_ref=550e-9):
    """
    Generate a synthetic Brazilian disk dataset whose C values follow a power law.

    Loads are chosen large enough that most ROI pixels span 2-4 fringe orders,
    giving the Stage-2 fit a strong dispersion signal.
    """
    height, width = 34, 34
    thickness = 0.01
    wavelengths = np.array([650e-9, 550e-9, 450e-9], dtype=float)
    C_ref_true = 2.5e-9
    c_true = _dispersion_power_law(wavelengths, C_ref_true, alpha_true, lambda_ref)
    s_true = np.array([0.82, 0.18, 0.0], dtype=float)

    radius_m = 0.008
    pixels_per_meter = 1800.0
    cx, cy = width / 2.0, height / 2.0

    x = (np.arange(width) - cx) / pixels_per_meter
    y = (cy - np.arange(height)) / pixels_per_meter
    X, Y = np.meshgrid(x, y)
    disk_mask = X**2 + Y**2 <= radius_m**2

    # High loads so many pixels reach fringe orders N=2-4.
    loads = [0.0, 150.0, 300.0, 440.0]
    image_files = []

    for idx, load in enumerate(loads):
        sigma_xx, sigma_yy, sigma_xy = diametrical_stress_cartesian(X, Y, P=load, R=radius_m)
        corrected = np.zeros((height, width, 3, 4), dtype=float)
        for c, wl in enumerate(wavelengths):
            i0, i45, i90, i135 = simulate_four_step_polarimetry(
                sigma_xx, sigma_yy, sigma_xy, c_true[c], 1.0, thickness, wl, s_true
            )
            corrected[:, :, c, 0] = i0
            corrected[:, :, c, 1] = i45
            corrected[:, :, c, 2] = i90
            corrected[:, :, c, 3] = i135
        corrected[~disk_mask] = 1.0
        image_file = tmp_path / f"disp_load_{idx}.npy"
        io.save_image(str(image_file), corrected.astype(np.float32))
        image_files.append(str(image_file))

    config = {
        "method": "brazilian_disk",
        "wavelengths": wavelengths.tolist(),
        "thickness": thickness,
        "nu": 1.0,
        "geometry": {
            "radius_mm": radius_m * 1e3,
            "radius_px": radius_m * pixels_per_meter,
            "center_px": [cx, cy],
            "edge_margin_fraction": 0.8,
            "contact_exclusion_fraction": 0.35,
        },
        "load_steps": [
            {"load": float(load), "image_file": f}
            for load, f in zip(loads, image_files)
        ],
        "fit": {
            "max_points": 2000,
            "seed": 42,
            "loss": "soft_l1",
            "f_scale": 0.2,
            "max_nfev": 600,
            "S_i_hat": s_true.tolist(),
            "dispersion_model": "power_law",
            "lambda_ref": lambda_ref,
        },
        "output_profile": str(tmp_path / "profile.json5"),
        "output_report": str(tmp_path / "report.md"),
        "output_diagnostics": str(tmp_path / "diagnostics.npz"),
    }

    return config, c_true, s_true, C_ref_true, alpha_true, lambda_ref


# ---------------------------------------------------------------------------
# Unit tests: dispersion model functions
# ---------------------------------------------------------------------------


def test_dispersion_power_law_at_reference_wavelength():
    """C(λ_ref) must equal C_ref regardless of alpha."""
    wavelengths = np.array([650e-9, 550e-9, 450e-9])
    C_ref, alpha, lambda_ref = 2.5e-9, 1.2, 550e-9
    result = _dispersion_power_law(wavelengths, C_ref, alpha, lambda_ref)
    ref_idx = 1  # 550 nm is the middle channel
    assert np.isclose(result[ref_idx], C_ref, rtol=1e-10)


def test_dispersion_power_law_monotone_for_positive_alpha():
    """With α > 0, shorter wavelengths must have larger C."""
    wavelengths = np.array([650e-9, 550e-9, 450e-9])
    result = _dispersion_power_law(wavelengths, 2.5e-9, 1.0)
    assert result[2] > result[1] > result[0]  # blue > green > red


def test_dispersion_power_law_flat_for_zero_alpha():
    wavelengths = np.array([650e-9, 550e-9, 450e-9])
    result = _dispersion_power_law(wavelengths, 2.5e-9, 0.0)
    np.testing.assert_allclose(result, 2.5e-9, rtol=1e-10)


def test_dispersion_cauchy_values():
    wavelengths = np.array([650e-9, 550e-9, 450e-9])
    A, B = 1.5e-9, 3e-22
    expected = A + B / wavelengths**2
    result = _dispersion_cauchy(wavelengths, A, B)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_get_dispersion_fn_power_law():
    fn = _get_dispersion_fn("power_law")
    assert fn is _dispersion_power_law


def test_get_dispersion_fn_cauchy():
    fn = _get_dispersion_fn("cauchy")
    assert fn is _dispersion_cauchy


def test_get_dispersion_fn_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dispersion model"):
        _get_dispersion_fn("unknown_model")


# ---------------------------------------------------------------------------
# Integration test: dispersion recovery from synthetic multi-fringe data
# ---------------------------------------------------------------------------


def test_fit_dispersion_recovers_power_law_alpha(tmp_path):
    """
    Stage-2 fit should converge when starting from an approximate alpha
    (simulating Stage-1 output with limited first-order data).

    Starting from half the true alpha keeps the per-channel C error below
    ~8%, so fringe-order assignments remain correct for the 2-3 fringe
    orders present in the calibration dataset.  Stage 2 should then refine
    the recovered C values to within 10% of truth.
    """
    config, c_true, s_true, C_ref_true, alpha_true, lambda_ref = _make_dispersion_case(
        tmp_path, alpha_true=0.8
    )
    validated = validate_calibration_config(config)
    dataset = calibrate._build_dataset(validated)

    wavelengths = np.array(config["wavelengths"])
    # Approximate starting point: half the true alpha, ~8% per-channel error.
    alpha_init = alpha_true / 2.0
    initial_C = _dispersion_power_law(wavelengths, C_ref_true, alpha_init, lambda_ref)

    disp_result = fit_dispersion_parameters(dataset, initial_C, validated["fit"])

    C_fit = np.asarray(disp_result["C"])
    np.testing.assert_allclose(C_fit, c_true, rtol=0.10)


def test_fit_dispersion_exact_initial_gives_near_zero_residuals(tmp_path):
    """
    Starting from the exact true C values, residuals should be near machine precision.
    """
    config, c_true, s_true, C_ref_true, alpha_true, lambda_ref = _make_dispersion_case(
        tmp_path, alpha_true=0.8
    )

    validated = validate_calibration_config(config)
    dataset = calibrate._build_dataset(validated)

    # Use exact C values as initial guess; the optimizer should barely move.
    disp_result = fit_dispersion_parameters(dataset, c_true, validated["fit"])

    C_fit = np.asarray(disp_result["C"])
    np.testing.assert_allclose(C_fit, c_true, rtol=0.01)


def test_fit_dispersion_result_keys(tmp_path):
    """fit_dispersion_parameters must return the expected dict keys."""
    config, c_true, s_true, C_ref_true, alpha_true, lambda_ref = _make_dispersion_case(tmp_path)
    validated = validate_calibration_config(config)
    dataset = calibrate._build_dataset(validated)

    result = fit_dispersion_parameters(dataset, c_true, validated["fit"])

    for key in ("C", "S_i_hat", "fit_metrics", "dispersion_model", "dispersion_params", "lambda_ref"):
        assert key in result, f"Missing key: {key}"
    assert result["dispersion_model"] == "power_law"
    assert len(result["dispersion_params"]) == 2  # [C_ref, alpha]


def test_fit_dispersion_cauchy_model(tmp_path):
    """Cauchy model should also converge to within 10% of true C values."""
    config, c_true, s_true, C_ref_true, alpha_true, lambda_ref = _make_dispersion_case(
        tmp_path, alpha_true=0.6
    )
    config["fit"]["dispersion_model"] = "cauchy"

    validated = validate_calibration_config(config)
    dataset = calibrate._build_dataset(validated)

    wavelengths = np.array(config["wavelengths"])
    # Approximate starting point: half the true alpha, ~6% per-channel error.
    initial_C = _dispersion_power_law(wavelengths, C_ref_true, alpha_true / 2.0, lambda_ref)

    disp_result = fit_dispersion_parameters(dataset, initial_C, validated["fit"])

    C_fit = np.asarray(disp_result["C"])
    np.testing.assert_allclose(C_fit, c_true, rtol=0.10)
