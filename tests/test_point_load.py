import numpy as np

from photoelastimetry.generate.point_load import (
    boussinesq_stress_cartesian,
    generate_synthetic_boussinesq,
    simulate_four_step_polarimetry,
)


def _symmetric_grid(n=61, span=0.02, depth_min=1e-3, depth_max=0.02):
    x = np.linspace(-span, span, n)
    y = np.linspace(depth_min, depth_max, n)
    return np.meshgrid(x, y)


def test_simulate_four_step_zero_stress_matches_input_stokes():
    s_i_hat = np.array([0.2, -0.3, 0.0])
    i0, i45, i90, i135 = simulate_four_step_polarimetry(
        0.0,
        0.0,
        0.0,
        C=2e-12,
        nu=1.0,
        L=0.01,
        wavelength=532e-9,
        S_i_hat=s_i_hat,
        I0=2.0,
    )

    # With zero stress, Mueller matrix is identity.
    assert np.isclose(i0, 2.0 * (1.0 + s_i_hat[0]) / 2)
    assert np.isclose(i45, 2.0 * (1.0 + s_i_hat[1]) / 2)
    assert np.isclose(i90, 2.0 * (1.0 - s_i_hat[0]) / 2)
    assert np.isclose(i135, 2.0 * (1.0 - s_i_hat[1]) / 2)


def test_boussinesq_even_odd_symmetry():
    x = np.linspace(-0.02, 0.02, 81)
    y = np.linspace(0.001, 0.02, 61)
    x_mid = len(x) // 2
    X, Y = np.meshgrid(x, y)

    s_xx, s_yy, t_xy = boussinesq_stress_cartesian(X, Y, P=120.0, nu_poisson=0.3)

    left_xx = s_xx[:, :x_mid]
    right_xx = np.flip(s_xx[:, x_mid + 1 :], axis=1)
    left_yy = s_yy[:, :x_mid]
    right_yy = np.flip(s_yy[:, x_mid + 1 :], axis=1)
    left_xy = t_xy[:, :x_mid]
    right_xy = np.flip(t_xy[:, x_mid + 1 :], axis=1)

    assert np.allclose(left_xx, right_xx, rtol=1e-6, atol=1e-8)
    assert np.allclose(left_yy, right_yy, rtol=1e-6, atol=1e-8)
    assert np.allclose(left_xy, -right_xy, rtol=1e-6, atol=1e-8)


def test_boussinesq_zero_above_surface():
    x = np.linspace(-0.02, 0.02, 41)
    y = np.linspace(-0.01, 0.02, 41)
    X, Y = np.meshgrid(x, y)

    s_xx, s_yy, t_xy = boussinesq_stress_cartesian(X, Y, P=100.0, nu_poisson=0.25)
    above = Y < 0

    assert np.all(s_xx[above] == 0)
    assert np.all(s_yy[above] == 0)
    assert np.all(t_xy[above] == 0)


def test_boussinesq_scales_linearly_with_load():
    X, Y = _symmetric_grid()

    s_xx_1, s_yy_1, t_xy_1 = boussinesq_stress_cartesian(X, Y, P=50.0, nu_poisson=0.3)
    s_xx_2, s_yy_2, t_xy_2 = boussinesq_stress_cartesian(X, Y, P=175.0, nu_poisson=0.3)
    scale = 175.0 / 50.0

    assert np.allclose(s_xx_2, s_xx_1 * scale, rtol=1e-6, atol=1e-6)
    assert np.allclose(s_yy_2, s_yy_1 * scale, rtol=1e-6, atol=1e-6)
    assert np.allclose(t_xy_2, t_xy_1 * scale, rtol=1e-6, atol=1e-6)


def test_generate_synthetic_boussinesq_mask_and_principal_invariants():
    X, Y = _symmetric_grid(n=45)
    mask = Y >= 0.004

    wavelengths = np.array([650e-9, 550e-9, 450e-9])
    c_values = np.array([2.0e-12, 2.2e-12, 2.5e-12])

    synthetic, principal_diff, theta_p, s_xx, s_yy, t_xy = generate_synthetic_boussinesq(
        X,
        Y,
        P=80.0,
        nu_poisson=0.3,
        S_i_hat=np.array([1.0, 0.0, 0.0]),
        mask=mask,
        wavelengths_nm=wavelengths,
        thickness=0.01,
        C=c_values,
        polarisation_efficiency=1.0,
    )

    assert synthetic.shape == (45, 45, 3, 4)
    assert np.all(np.isnan(s_xx[~mask]))
    assert np.all(np.isnan(s_yy[~mask]))
    assert np.all(np.isnan(t_xy[~mask]))

    expected_diff = np.sqrt((s_xx - s_yy) ** 2 + 4 * t_xy**2)
    expected_theta = 0.5 * np.arctan2(2 * t_xy, s_xx - s_yy)

    assert np.allclose(principal_diff[mask], expected_diff[mask], rtol=1e-8, atol=1e-10)
    assert np.allclose(theta_p[mask], expected_theta[mask], rtol=1e-8, atol=1e-10)


def test_generate_synthetic_boussinesq_channel_dependence_on_optics():
    X, Y = _symmetric_grid(n=31)
    mask = Y >= 0.0

    wavelengths = np.array([650e-9, 550e-9, 450e-9])
    c_values = np.array([1.9e-12, 2.3e-12, 2.9e-12])

    synthetic, *_ = generate_synthetic_boussinesq(
        X,
        Y,
        P=150.0,
        nu_poisson=0.35,
        S_i_hat=np.array([1.0, 0.0, 0.0]),
        mask=mask,
        wavelengths_nm=wavelengths,
        thickness=0.01,
        C=c_values,
        polarisation_efficiency=1.0,
    )

    # At nonzero stress, differing (C, wavelength) should produce different channel responses.
    mean_abs_01 = np.mean(np.abs(synthetic[:, :, 0, :] - synthetic[:, :, 1, :]))
    mean_abs_12 = np.mean(np.abs(synthetic[:, :, 1, :] - synthetic[:, :, 2, :]))

    assert mean_abs_01 > 1e-6
    assert mean_abs_12 > 1e-6
