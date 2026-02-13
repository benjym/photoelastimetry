import numpy as np
import pytest

from photoelastimetry.bspline import BSplineAiry, BSplineScalar


def _finite_difference_grad(objective, coeffs, idx, eps=1e-6):
    c_plus = coeffs.copy()
    c_minus = coeffs.copy()
    c_plus[idx] += eps
    c_minus[idx] -= eps
    return (objective(c_plus) - objective(c_minus)) / (2 * eps)


def test_project_stress_gradients_matches_finite_difference():
    rng = np.random.default_rng(0)
    bs = BSplineAiry((14, 16), knot_spacing=4, degree=3)

    coeffs = rng.normal(scale=1e-2, size=bs.n_coeffs)
    grad_xx = rng.normal(size=(14, 16))
    grad_yy = rng.normal(size=(14, 16))
    grad_xy = rng.normal(size=(14, 16))

    def objective(c):
        s_xx, s_yy, s_xy = bs.get_stress_fields(c)
        return np.sum(s_xx * grad_xx + s_yy * grad_yy + s_xy * grad_xy)

    analytic = bs.project_stress_gradients(grad_xx, grad_yy, grad_xy)

    for idx in [0, bs.n_coeffs // 2, bs.n_coeffs - 1]:
        numeric = _finite_difference_grad(objective, coeffs, idx)
        assert np.isclose(analytic[idx], numeric, rtol=1e-4, atol=1e-4)


def test_project_scalar_gradients_matches_finite_difference():
    rng = np.random.default_rng(1)
    bs = BSplineScalar((12, 13), knot_spacing=4, degree=3)

    coeffs = rng.normal(scale=1e-2, size=bs.n_coeffs)
    grad_p = rng.normal(size=(12, 13))
    grad_dx = rng.normal(size=(12, 13))
    grad_dy = rng.normal(size=(12, 13))

    def objective(c):
        p, dp_dx, dp_dy = bs.get_scalar_fields(c)
        return np.sum(p * grad_p + dp_dx * grad_dx + dp_dy * grad_dy)

    analytic = bs.project_scalar_gradients(grad_p, grad_dx, grad_dy)

    for idx in [0, bs.n_coeffs // 3, bs.n_coeffs - 1]:
        numeric = _finite_difference_grad(objective, coeffs, idx)
        assert np.isclose(analytic[idx], numeric, rtol=1e-4, atol=1e-4)


def test_fit_stress_field_handles_nan_targets():
    bs = BSplineAiry((12, 12), knot_spacing=4, degree=3)

    stress = np.zeros((12, 12, 3), dtype=float)
    stress[3:8, 4:9, :] = np.nan

    coeffs = bs.fit_stress_field(stress)
    s_xx, s_yy, s_xy = bs.get_stress_fields(coeffs)

    valid = ~np.isnan(stress[..., 0])
    assert np.isfinite(coeffs).all()
    assert np.max(np.abs(s_xx[valid])) < 1e-6
    assert np.max(np.abs(s_yy[valid])) < 1e-6
    assert np.max(np.abs(s_xy[valid])) < 1e-6


def test_fit_scalar_field_invalid_shape_raises():
    bs = BSplineScalar((10, 11), knot_spacing=4, degree=3)

    with pytest.raises(ValueError, match="scalar_field shape must be"):
        bs.fit_scalar_field(np.zeros((10, 10)))


def test_fit_scalar_field_all_invalid_mask_returns_zeros():
    bs = BSplineScalar((9, 9), knot_spacing=3, degree=3)
    scalar = np.full((9, 9), np.nan)
    mask = np.zeros((9, 9), dtype=bool)

    coeffs = bs.fit_scalar_field(scalar, mask=mask)

    assert np.allclose(coeffs, 0.0)
