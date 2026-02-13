import numpy as np

from photoelastimetry.generate.strip_load import generate_synthetic_strip_load, strip_load_stress_cartesian


def _grid(n=61, span=0.05, depth_min=0.001, depth_max=0.04):
    x = np.linspace(-span, span, n)
    y = np.linspace(depth_min, depth_max, n)
    return np.meshgrid(x, y)


def test_strip_load_even_odd_symmetry():
    X, Y = _grid(n=81)

    s_xx, s_yy, t_xy = strip_load_stress_cartesian(X, Y, p=1500.0, a=0.01)

    mid = X.shape[1] // 2
    assert np.allclose(s_xx[:, :mid], np.flip(s_xx[:, mid + 1 :], axis=1), rtol=1e-8, atol=1e-8)
    assert np.allclose(s_yy[:, :mid], np.flip(s_yy[:, mid + 1 :], axis=1), rtol=1e-8, atol=1e-8)
    assert np.allclose(t_xy[:, :mid], -np.flip(t_xy[:, mid + 1 :], axis=1), rtol=1e-8, atol=1e-8)


def test_strip_load_zero_above_surface_and_scaling():
    x = np.linspace(-0.04, 0.04, 51)
    y = np.linspace(-0.01, 0.03, 51)
    X, Y = np.meshgrid(x, y)

    s_xx_1, s_yy_1, t_xy_1 = strip_load_stress_cartesian(X, Y, p=500.0, a=0.008)
    s_xx_2, s_yy_2, t_xy_2 = strip_load_stress_cartesian(X, Y, p=1250.0, a=0.008)

    above = Y < 0
    assert np.all(s_xx_1[above] == 0)
    assert np.all(s_yy_1[above] == 0)
    assert np.all(t_xy_1[above] == 0)

    scale = 1250.0 / 500.0
    below = Y >= 0
    assert np.allclose(s_xx_2[below], s_xx_1[below] * scale, rtol=1e-8, atol=1e-8)
    assert np.allclose(s_yy_2[below], s_yy_1[below] * scale, rtol=1e-8, atol=1e-8)
    assert np.allclose(t_xy_2[below], t_xy_1[below] * scale, rtol=1e-8, atol=1e-8)


def test_generate_synthetic_strip_load_mask_and_invariants():
    X, Y = _grid(n=45, span=0.03, depth_min=0.0, depth_max=0.03)
    mask = Y >= 0.004

    wavelengths = np.array([650e-9, 550e-9, 450e-9])
    c_values = np.array([2.0e-12, 2.2e-12, 2.5e-12])

    synthetic, principal_diff, theta_p, s_xx, s_yy, t_xy = generate_synthetic_strip_load(
        X,
        Y,
        p=1200.0,
        a=0.007,
        S_i_hat=np.array([1.0, 0.0, 0.0]),
        mask=mask,
        wavelengths_nm=wavelengths,
        thickness=0.01,
        C=c_values,
        polarisation_efficiency=1.0,
    )

    expected_diff = np.sqrt((s_xx - s_yy) ** 2 + 4 * t_xy**2)
    expected_theta = 0.5 * np.arctan2(2 * t_xy, s_xx - s_yy)

    assert synthetic.shape == (45, 45, 3, 4)
    assert np.all(np.isnan(s_xx[~mask]))
    assert np.all(np.isnan(s_yy[~mask]))
    assert np.all(np.isnan(t_xy[~mask]))
    assert np.allclose(principal_diff[mask], expected_diff[mask], rtol=1e-10, atol=1e-12)
    assert np.allclose(theta_p[mask], expected_theta[mask], rtol=1e-10, atol=1e-12)
