import numpy as np

from photoelastimetry.generate.disk import diametrical_stress_cartesian, generate_synthetic_brazil_test


def _disk_grid(radius=0.01, n=81):
    x = np.linspace(-radius, radius, n)
    y = np.linspace(-radius, radius, n)
    return np.meshgrid(x, y)


def test_diametrical_stress_symmetry_and_center_signs():
    r = 0.01
    X, Y = _disk_grid(radius=r, n=91)

    s_xx, s_yy, t_xy = diametrical_stress_cartesian(X, Y, P=1000.0, R=r)

    # Symmetry in x: sigma_xx, sigma_yy even; tau_xy odd.
    mid = X.shape[1] // 2
    assert np.allclose(s_xx[:, :mid], np.flip(s_xx[:, mid + 1 :], axis=1), rtol=2e-4, atol=1e-5)
    assert np.allclose(s_yy[:, :mid], np.flip(s_yy[:, mid + 1 :], axis=1), rtol=2e-4, atol=1e-5)
    assert np.allclose(t_xy[:, :mid], -np.flip(t_xy[:, mid + 1 :], axis=1), rtol=2e-4, atol=1e-5)

    # Field should be non-uniform and stress magnitudes should be finite away from singular locations.
    finite = np.isfinite(s_xx) & np.isfinite(s_yy) & np.isfinite(t_xy)
    assert np.std(s_xx[finite]) > 0
    assert np.std(s_yy[finite]) > 0


def test_diametrical_stress_linear_scaling_with_load():
    r = 0.008
    X, Y = _disk_grid(radius=r, n=61)

    s_xx_1, s_yy_1, t_xy_1 = diametrical_stress_cartesian(X, Y, P=200.0, R=r)
    s_xx_2, s_yy_2, t_xy_2 = diametrical_stress_cartesian(X, Y, P=500.0, R=r)
    scale = 500.0 / 200.0

    finite = np.isfinite(s_xx_1) & np.isfinite(s_xx_2) & np.isfinite(s_yy_1) & np.isfinite(s_yy_2)
    finite &= np.isfinite(t_xy_1) & np.isfinite(t_xy_2)

    assert np.allclose(s_xx_2[finite], s_xx_1[finite] * scale, rtol=1e-6, atol=1e-6)
    assert np.allclose(s_yy_2[finite], s_yy_1[finite] * scale, rtol=1e-6, atol=1e-6)
    assert np.allclose(t_xy_2[finite], t_xy_1[finite] * scale, rtol=1e-6, atol=1e-6)


def test_generate_synthetic_brazil_test_mask_and_principal_diff():
    r = 0.01
    X, Y = _disk_grid(radius=r, n=55)
    mask = np.sqrt(X**2 + Y**2) <= 0.92 * r

    wavelengths = np.array([650e-9, 550e-9, 450e-9])
    c_values = np.array([2.0e-12, 2.2e-12, 2.4e-12])

    synthetic, principal_diff, theta_p, s_xx, s_yy, t_xy = generate_synthetic_brazil_test(
        X,
        Y,
        P=600.0,
        R=r,
        S_i_hat=np.array([1.0, 0.0, 0.0]),
        mask=mask,
        wavelengths_nm=wavelengths,
        thickness=0.01,
        C=c_values,
        polarisation_efficiency=1.0,
    )

    assert synthetic.shape == (55, 55, 3, 4)
    assert np.all(np.isnan(s_xx[~mask]))
    assert np.all(np.isnan(s_yy[~mask]))
    assert np.all(np.isnan(t_xy[~mask]))

    expected_diff = np.sqrt((s_xx - s_yy) ** 2 + 4 * t_xy**2)
    expected_theta = 0.5 * np.arctan2(2 * t_xy, s_xx - s_yy)

    assert np.allclose(principal_diff[mask], expected_diff[mask], rtol=1e-8, atol=1e-10)
    assert np.allclose(theta_p[mask], expected_theta[mask], rtol=1e-8, atol=1e-10)


def test_generate_synthetic_brazil_test_channel_dependence_on_optics():
    r = 0.01
    X, Y = _disk_grid(radius=r, n=41)
    mask = np.sqrt(X**2 + Y**2) <= r

    wavelengths = np.array([650e-9, 540e-9, 430e-9])
    c_values = np.array([1.8e-12, 2.3e-12, 2.9e-12])

    synthetic, *_ = generate_synthetic_brazil_test(
        X,
        Y,
        P=900.0,
        R=r,
        S_i_hat=np.array([1.0, 0.0, 0.0]),
        mask=mask,
        wavelengths_nm=wavelengths,
        thickness=0.012,
        C=c_values,
        polarisation_efficiency=1.0,
    )

    valid = (
        np.isfinite(synthetic[:, :, 0, 0])
        & np.isfinite(synthetic[:, :, 1, 0])
        & np.isfinite(synthetic[:, :, 2, 0])
    )
    ch0 = synthetic[valid, 0, :]
    ch1 = synthetic[valid, 1, :]
    ch2 = synthetic[valid, 2, :]

    assert not np.allclose(ch0, ch1, rtol=1e-8, atol=1e-12)
    assert not np.allclose(ch1, ch2, rtol=1e-8, atol=1e-12)
