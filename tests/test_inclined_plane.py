import numpy as np

from photoelastimetry.generate.inclined_plane import (
    generate_synthetic_inclined_plane,
    inclined_stress_cartesian,
)


def _grid(width=0.1, height=0.08, n=40):
    x = np.linspace(0.0, width, n)
    y = np.linspace(0.0, height, n)
    return np.meshgrid(x, y)


def test_inclined_vertical_case_matches_lithostatic_relation():
    X, Y = _grid(n=35)
    rho = 2300.0
    g = 9.81
    k0 = 0.6

    s_xx, s_yy, t_xy = inclined_stress_cartesian(X, Y, rho=rho, g=g, theta_deg=0.0, K0=k0)

    assert np.allclose(t_xy, 0.0, atol=1e-12)
    assert np.allclose(s_yy, rho * g * Y, rtol=1e-12, atol=1e-12)
    assert np.allclose(s_xx, k0 * rho * g * Y, rtol=1e-12, atol=1e-12)


def test_inclined_stress_shear_sign_changes_with_angle_sign():
    X, Y = _grid(n=30)
    rho = 2500.0

    _, _, tau_pos = inclined_stress_cartesian(X, Y, rho=rho, g=9.81, theta_deg=25.0, K0=0.5)
    _, _, tau_neg = inclined_stress_cartesian(X, Y, rho=rho, g=9.81, theta_deg=-25.0, K0=0.5)

    assert np.allclose(tau_pos, -tau_neg, rtol=1e-12, atol=1e-12)


def test_inclined_stress_respects_k0_scaling_for_sigma_xx():
    X, Y = _grid(n=32)
    rho = 2100.0
    g = 9.81
    theta = 30.0

    s_xx_low, s_yy, _ = inclined_stress_cartesian(X, Y, rho=rho, g=g, theta_deg=theta, K0=0.35)
    s_xx_high, _, _ = inclined_stress_cartesian(X, Y, rho=rho, g=g, theta_deg=theta, K0=0.8)

    assert np.allclose(s_xx_low, 0.35 * s_yy, rtol=1e-12, atol=1e-12)
    assert np.allclose(s_xx_high, 0.8 * s_yy, rtol=1e-12, atol=1e-12)


def test_generate_synthetic_inclined_plane_principal_invariants():
    X, Y = _grid(n=28)

    wavelengths = np.array([650e-9, 550e-9, 450e-9])
    c_values = np.array([1.8e-12, 2.0e-12, 2.4e-12])

    synthetic, principal_diff, theta_p, s_xx, s_yy, t_xy = generate_synthetic_inclined_plane(
        X,
        Y,
        rho=2400.0,
        g=9.81,
        theta_deg=20.0,
        K0=0.55,
        S_i_hat=np.array([1.0, 0.0, 0.0]),
        wavelengths_nm=wavelengths,
        thickness=0.01,
        C=c_values,
    )

    expected_diff = np.sqrt((s_xx - s_yy) ** 2 + 4 * t_xy**2)
    expected_theta = 0.5 * np.arctan2(2 * t_xy, s_xx - s_yy)

    assert synthetic.shape == (28, 28, 3, 4)
    assert np.allclose(principal_diff, expected_diff, rtol=1e-10, atol=1e-12)
    assert np.allclose(theta_p, expected_theta, rtol=1e-10, atol=1e-12)


def test_generate_synthetic_inclined_plane_channel_dependence_on_optics():
    X, Y = _grid(n=30)
    wavelengths = np.array([650e-9, 560e-9, 470e-9])
    c_values = np.array([1.5e-12, 2.2e-12, 3.0e-12])

    synthetic, *_ = generate_synthetic_inclined_plane(
        X,
        Y,
        rho=2600.0,
        g=9.81,
        theta_deg=35.0,
        K0=0.5,
        S_i_hat=np.array([1.0, 0.0, 0.0]),
        wavelengths_nm=wavelengths,
        thickness=0.012,
        C=c_values,
    )

    ch0 = synthetic[:, :, 0, :]
    ch1 = synthetic[:, :, 1, :]
    ch2 = synthetic[:, :, 2, :]

    assert not np.allclose(ch0, ch1, rtol=1e-8, atol=1e-12)
    assert not np.allclose(ch1, ch2, rtol=1e-8, atol=1e-12)
