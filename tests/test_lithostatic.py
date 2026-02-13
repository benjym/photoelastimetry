import numpy as np

from photoelastimetry.generate.lithostatic import generate_synthetic_lithostatic, lithostatic_stress_cartesian


def _grid(width=0.1, depth=0.08, n=50):
    x = np.linspace(0.0, width, n)
    y = np.linspace(-0.02, depth, n)
    return np.meshgrid(x, y)


def test_lithostatic_exact_formula_and_depth_clipping():
    X, Y = _grid(n=41)
    rho = 2400.0
    g = 9.81
    k0 = 0.55

    s_xx, s_yy, t_xy = lithostatic_stress_cartesian(X, Y, rho=rho, g=g, K0=k0)

    expected_syy = rho * g * np.maximum(0.0, Y)
    expected_sxx = k0 * expected_syy

    assert np.allclose(s_yy, expected_syy, rtol=1e-12, atol=1e-12)
    assert np.allclose(s_xx, expected_sxx, rtol=1e-12, atol=1e-12)
    assert np.allclose(t_xy, 0.0, atol=1e-12)


def test_lithostatic_k0_proportionality_for_multiple_values():
    X, Y = _grid(n=31)
    rho = 2200.0

    for k0 in [0.3, 0.5, 0.7, 1.0]:
        s_xx, s_yy, _ = lithostatic_stress_cartesian(X, Y, rho=rho, g=9.81, K0=k0)
        assert np.allclose(s_xx, k0 * s_yy, rtol=1e-12, atol=1e-12)


def test_generate_synthetic_lithostatic_output_and_invariants():
    x = np.linspace(0.0, 0.06, 36)
    y = np.linspace(0.0, 0.05, 36)
    X, Y = np.meshgrid(x, y)

    wavelengths = np.array([650e-9, 550e-9, 450e-9])
    c_values = np.array([2.0e-12, 2.2e-12, 2.4e-12])

    synthetic, principal_diff, theta_p, s_xx, s_yy, t_xy = generate_synthetic_lithostatic(
        X,
        Y,
        rho=2500.0,
        g=9.81,
        K0=0.5,
        S_i_hat=np.array([1.0, 0.0, 0.0]),
        wavelengths_nm=wavelengths,
        thickness=0.01,
        C=c_values,
    )

    expected_diff = np.abs(s_xx - s_yy)
    expected_theta = 0.5 * np.arctan2(2 * t_xy, s_xx - s_yy)

    assert synthetic.shape == (36, 36, 3, 4)
    assert np.allclose(t_xy, 0.0, atol=1e-12)
    assert np.allclose(principal_diff, expected_diff, rtol=1e-12, atol=1e-12)
    assert np.allclose(theta_p, expected_theta, rtol=1e-12, atol=1e-12)


def test_generate_synthetic_lithostatic_channel_dependence_on_optics():
    x = np.linspace(0.0, 0.04, 30)
    y = np.linspace(0.0, 0.03, 30)
    X, Y = np.meshgrid(x, y)

    wavelengths = np.array([650e-9, 560e-9, 470e-9])
    c_values = np.array([1.5e-12, 2.2e-12, 3.0e-12])

    synthetic, *_ = generate_synthetic_lithostatic(
        X,
        Y,
        rho=2600.0,
        g=9.81,
        K0=0.4,
        S_i_hat=np.array([0.2, 0.4, 0.3]),
        wavelengths_nm=wavelengths,
        thickness=0.01,
        C=c_values,
    )

    ch0 = synthetic[:, :, 0, :]
    ch1 = synthetic[:, :, 1, :]
    ch2 = synthetic[:, :, 2, :]

    assert not np.allclose(ch0, ch1, rtol=1e-8, atol=1e-12)
    assert not np.allclose(ch1, ch2, rtol=1e-8, atol=1e-12)
