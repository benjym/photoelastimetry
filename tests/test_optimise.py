"""Tests for the optimise solver."""

import numpy as np
import pytest

from photoelastimetry.optimise import recover_mean_stress, stress_to_principal_invariants


def _wrap_half_pi(angle_diff):
    """Wrap angle differences to [-pi/2, pi/2)."""
    return (angle_diff + np.pi / 2) % np.pi - np.pi / 2


def test_recover_mean_stress_keeps_deviatoric_invariants():
    """The solver should keep input (delta_sigma, theta) exactly in reconstructed stresses."""
    rng = np.random.default_rng(0)
    h, w = 28, 34
    delta_sigma = np.abs(rng.normal(5e3, 2e3, size=(h, w))) + 10.0
    theta = rng.uniform(-np.pi / 2, np.pi / 2, size=(h, w))

    wrapper, coeffs = recover_mean_stress(
        delta_sigma,
        theta,
        knot_spacing=6,
        max_iterations=30,
        tolerance=1e-8,
        verbose=False,
    )

    s_xx, s_yy, s_xy = wrapper.get_stress_fields(coeffs)
    recovered_delta, recovered_theta = stress_to_principal_invariants(np.stack([s_xx, s_yy, s_xy], axis=-1))

    angle_err = _wrap_half_pi(recovered_theta - theta)

    assert np.max(np.abs(recovered_delta - delta_sigma)) < 1e-8
    assert np.max(np.abs(angle_err)) < 1e-10


def test_recover_mean_stress_uses_external_potential_gradient():
    """With zero deviatoric stress and linear potential V, solved pressure should match V up to a constant."""
    h, w = 30, 36
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    V = 2.5 * X - 1.25 * Y
    delta_sigma = np.zeros((h, w))
    theta = np.zeros((h, w))

    wrapper, coeffs = recover_mean_stress(
        delta_sigma,
        theta,
        knot_spacing=5,
        external_potential=V,
        max_iterations=200,
        tolerance=1e-10,
        verbose=False,
    )

    P, _, _ = wrapper.bspline_p.get_scalar_fields(coeffs)

    P_centered = P - np.mean(P)
    V_centered = V - np.mean(V)

    rmse = np.sqrt(np.mean((P_centered - V_centered) ** 2))
    assert rmse < 1e-3


def test_recover_mean_stress_zero_mean_gauge_without_boundary():
    """Without pressure BCs, solver should center pressure around zero."""
    rng = np.random.default_rng(42)
    h, w = 24, 24
    delta_sigma = np.abs(rng.normal(1e4, 4e3, size=(h, w))) + 5.0
    theta = rng.uniform(-np.pi / 2, np.pi / 2, size=(h, w))

    wrapper, coeffs = recover_mean_stress(
        delta_sigma,
        theta,
        knot_spacing=6,
        max_iterations=120,
        tolerance=1e-9,
        verbose=False,
    )

    P, _, _ = wrapper.bspline_p.get_scalar_fields(coeffs)
    assert abs(np.mean(P)) < 1e-3


def test_recover_mean_stress_rejects_external_potential_shape_mismatch():
    delta_sigma = np.zeros((8, 9))
    theta = np.zeros((8, 9))

    with pytest.raises(ValueError, match="external_potential must have shape"):
        recover_mean_stress(delta_sigma, theta, external_potential=np.zeros((8, 8)), verbose=False)


def test_recover_mean_stress_boundary_conditions_anchor_pressure_level():
    h, w = 22, 24
    delta_sigma = np.zeros((h, w))
    theta = np.zeros((h, w))

    boundary_mask = np.zeros((h, w), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True

    xx_target = np.full((h, w), np.nan)
    yy_target = np.full((h, w), np.nan)
    xx_target[boundary_mask] = 2.5e4
    yy_target[boundary_mask] = 2.5e4

    wrapper, coeffs = recover_mean_stress(
        delta_sigma,
        theta,
        knot_spacing=5,
        boundary_mask=boundary_mask,
        boundary_values={"xx": xx_target, "yy": yy_target},
        boundary_weight=2.0,
        max_iterations=160,
        tolerance=1e-10,
        verbose=False,
    )

    P, _, _ = wrapper.bspline_p.get_scalar_fields(coeffs)

    mean_boundary_pressure = np.mean(P[boundary_mask])
    assert abs(mean_boundary_pressure - 2.5e4) < 50.0


def test_recover_mean_stress_with_regularisation_branch():
    rng = np.random.default_rng(11)
    h, w = 20, 20
    delta_sigma = np.abs(rng.normal(8e3, 2e3, size=(h, w)))
    theta = rng.uniform(-np.pi / 2, np.pi / 2, size=(h, w))

    wrapper, coeffs = recover_mean_stress(
        delta_sigma,
        theta,
        knot_spacing=5,
        regularisation_weight=1e-3,
        max_iterations=80,
        tolerance=1e-8,
        verbose=False,
    )

    s_xx, s_yy, s_xy = wrapper.get_stress_fields(coeffs)
    assert np.isfinite(coeffs).all()
    assert np.isfinite(s_xx).all()
    assert np.isfinite(s_yy).all()
    assert np.isfinite(s_xy).all()


def test_recover_mean_stress_handles_nan_inputs_with_masking():
    h, w = 18, 19
    delta_sigma = np.full((h, w), 4e3, dtype=float)
    theta = np.zeros((h, w), dtype=float)

    delta_sigma[3:6, 5:9] = np.nan
    theta[10:13, 2:7] = np.nan

    wrapper, coeffs = recover_mean_stress(
        delta_sigma,
        theta,
        knot_spacing=4,
        max_iterations=100,
        tolerance=1e-8,
        verbose=False,
    )

    P, _, _ = wrapper.bspline_p.get_scalar_fields(coeffs)
    assert np.isfinite(coeffs).all()
    assert np.isfinite(P).all()
