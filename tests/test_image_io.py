import numpy as np
import pytest

from photoelastimetry.image import (
    AoLP,
    DoLP,
    compute_principal_angle,
    compute_retardance,
    mueller_matrix,
    mueller_matrix_sensitivity,
    predict_stokes,
    simulate_four_step_polarimetry,
)


def test_compute_retardance_matches_formula_scalar_and_array():
    sigma_xx = np.array([2.0e6, 1.0e6])
    sigma_yy = np.array([-1.0e6, 0.5e6])
    sigma_xy = np.array([0.25e6, 0.75e6])

    c = 2.2e-12
    nu = 1.0
    thickness = 0.01
    wavelength = 532e-9

    delta = compute_retardance(sigma_xx, sigma_yy, sigma_xy, c, nu, thickness, wavelength)
    expected = (2 * np.pi * c * nu * thickness / wavelength) * np.sqrt(
        (sigma_xx - sigma_yy) ** 2 + 4 * sigma_xy**2
    )

    assert np.allclose(delta, expected)
    assert np.all(delta >= 0)


def test_compute_principal_angle_special_cases():
    assert np.isclose(compute_principal_angle(1.0, 1.0, 1.0), np.pi / 4)
    assert np.isclose(compute_principal_angle(2.0, 1.0, 0.0), 0.0)
    assert compute_principal_angle(2.0, 1.0, -1.0) < 0


def test_mueller_matrix_identity_and_vectorized_shape():
    m_identity = mueller_matrix(0.0, 0.0)
    assert m_identity.shape == (4, 4)
    assert np.allclose(m_identity, np.eye(4), atol=1e-12)

    theta = np.linspace(-0.3, 0.4, 6).reshape(2, 3)
    delta = np.linspace(0.0, np.pi, 6).reshape(2, 3)
    m = mueller_matrix(theta, delta)

    assert m.shape == (2, 3, 4, 4)
    assert np.all(np.isfinite(m))
    assert np.allclose(m[..., 0, 0], 1.0)


def test_mueller_matrix_sensitivity_matches_finite_difference():
    theta = 0.31
    delta = 1.07
    eps = 1e-7

    dtheta_analytic, ddelta_analytic = mueller_matrix_sensitivity(theta, delta)

    dtheta_numeric = (mueller_matrix(theta + eps, delta) - mueller_matrix(theta - eps, delta)) / (2 * eps)
    ddelta_numeric = (mueller_matrix(theta, delta + eps) - mueller_matrix(theta, delta - eps)) / (2 * eps)

    assert np.allclose(dtheta_analytic, dtheta_numeric, rtol=1e-5, atol=1e-6)
    assert np.allclose(ddelta_analytic, ddelta_numeric, rtol=1e-5, atol=1e-6)


def test_predict_stokes_vectorized_matches_pixelwise_loop():
    rng = np.random.default_rng(7)
    sigma_xx = rng.normal(loc=1e6, scale=2e5, size=(4, 5))
    sigma_yy = rng.normal(loc=0.5e6, scale=2e5, size=(4, 5))
    sigma_xy = rng.normal(loc=0.2e6, scale=1e5, size=(4, 5))

    kwargs = dict(C=2.0e-12, nu=1.0, L=0.01, wavelength=550e-9, S_i_hat=np.array([1.0, 0.0, 0.0]))

    vec = predict_stokes(sigma_xx, sigma_yy, sigma_xy, **kwargs)

    ref = np.zeros((4, 5, 2), dtype=float)
    for i in range(4):
        for j in range(5):
            ref[i, j] = predict_stokes(
                float(sigma_xx[i, j]), float(sigma_yy[i, j]), float(sigma_xy[i, j]), **kwargs
            )

    assert vec.shape == (4, 5, 2)
    assert np.allclose(vec, ref, rtol=1e-10, atol=1e-10)


def test_predict_stokes_invalid_si_hat_length_raises():
    with pytest.raises(ValueError, match="S_i_hat must have length 2 or 3"):
        predict_stokes(1.0, 0.0, 0.0, C=2e-12, nu=1.0, L=0.01, wavelength=532e-9, S_i_hat=[1.0])


def test_simulate_four_step_invalid_si_hat_length_raises():
    with pytest.raises(ValueError, match="S_i_hat must have 2 or 3 elements"):
        simulate_four_step_polarimetry(
            1.0,
            0.0,
            0.0,
            C=2e-12,
            nu=1.0,
            L=0.01,
            wavelength=532e-9,
            S_i_hat=[1.0],
        )


def test_dolp_and_aolp_deterministic_pattern():
    image = np.zeros((1, 1, 1, 4), dtype=float)
    image[0, 0, 0, :] = [3.0, 1.0, 2.0, 0.5]

    dolp = DoLP(image)
    aolp = AoLP(image)

    q = 3.0 - 1.0
    u = 2.0 - 0.5
    i = 3.0 + 1.0 + 2.0 + 0.5

    assert np.isclose(dolp[0, 0, 0], np.sqrt(q**2 + u**2) / i)
    assert np.isclose(aolp[0, 0, 0], 0.5 * np.arctan2(u, q))


def test_simulate_four_step_vectorized_shape_and_bounds():
    sigma_xx = np.full((3, 4), 1.0e6)
    sigma_yy = np.full((3, 4), 0.5e6)
    sigma_xy = np.full((3, 4), 0.2e6)

    i0, i45, i90, i135 = simulate_four_step_polarimetry(
        sigma_xx,
        sigma_yy,
        sigma_xy,
        C=2e-12,
        nu=1.0,
        L=0.01,
        wavelength=650e-9,
        S_i_hat=[1.0, 0.0],
        I0=1.0,
    )

    for arr in (i0, i45, i90, i135):
        assert arr.shape == (3, 4)
        assert np.all(np.isfinite(arr))
