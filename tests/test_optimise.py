"""Tests for the optimise solver."""

import numpy as np

from photoelastimetry.optimise import recover_mean_stress, stress_to_principal_invariants


def _wrap_half_pi(angle_diff):
    """Wrap angle differences to [-pi/2, pi/2)."""
    return (angle_diff + np.pi / 2) % np.pi - np.pi / 2


def test_recover_mean_stress_keeps_deviatoric_invariants():
    """The solver should keep input (delta_sigma, theta) exactly in reconstructed stresses."""
    rng = np.random.default_rng(0)
    H, W = 28, 34
    delta_sigma = np.abs(rng.normal(5e3, 2e3, size=(H, W))) + 10.0
    theta = rng.uniform(-np.pi / 2, np.pi / 2, size=(H, W))

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
    """
    With zero deviatoric stress and linear potential V, solved pressure should match V up to a constant.
    """
    H, W = 30, 36
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    V = 2.5 * X - 1.25 * Y
    delta_sigma = np.zeros((H, W))
    theta = np.zeros((H, W))

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
    H, W = 24, 24
    delta_sigma = np.abs(rng.normal(1e4, 4e3, size=(H, W))) + 5.0
    theta = rng.uniform(-np.pi / 2, np.pi / 2, size=(H, W))

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
