import numpy as np
import pytest

from photoelastimetry.bspline import BSplineAiry


def fit_coefficients(bspline, phi_target):
    """
    Fit B-spline coefficients to a target 2D field phi_target.
    We solve the linear system:
    phi ~ By @ C @ Bx.T

    Since we can't easily invert the tensor product directly without Kronecker products
    (which would be huge), we can do it iteratively or separably if the target is separable.
    All our test cases (x^2, y^2, xy) are separable.

    BUT, for a general robust test utility, let's just use a simple alternating least squares
    or just solving for C in two steps:
    1. Solve for T = C @ Bx.T  =>  phi = By @ T  =>  T = pinv(By) @ phi
    2. Solve for C from T = C @ Bx.T => T.T = Bx @ C.T => C.T = pinv(Bx) @ T.T => C = (pinv(Bx) @ T.T).T
    """
    # Step 1: Solve By * T = phi  ->  T = pinv(By) * phi
    By_pinv = np.linalg.pinv(bspline.By)
    T = By_pinv @ phi_target

    # Step 2: Solve C * Bx.T = T  ->  Bx * C.T = T.T  ->  C.T = pinv(Bx) * T.T
    Bx_pinv = np.linalg.pinv(bspline.Bx)
    C_T = Bx_pinv @ T.T

    return C_T.T


def test_pure_shear_sign():
    """
    Test the sign of the shear stress calculation.
    Target Airy function: phi(x,y) = x * y (in image coordinates)

    Physics derivation:
        sigma_xy = - d^2(phi) / dx_phy dy_phy

    Coordinate transform (y_img positive down, y_phy positive up):
        dy_phy = -dy_img
        dx_phy = dx_img

    Therefore:
        sigma_xy = - d^2(phi) / dx_img (-dy_img)
                 = + d^2(phi) / dx_img dy_img

    For phi = x * y:
        d(phi)/dx = y
        d^2(phi)/dx dy = 1

    Expected Result: sigma_xy = +1 everywhere.
    """
    H, W = 50, 50
    bs = BSplineAiry((H, W), knot_spacing=5, degree=3)

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    phi = x * y

    C = fit_coefficients(bs, phi)

    s_xx, s_yy, s_xy = bs.get_stress_fields(C.flatten())

    # Check middle region to avoid boundary artifacts from splines
    margin = 10
    mid_s_xy = s_xy[margin:-margin, margin:-margin]

    # We expect +1.0
    print(f"Mean s_xy: {np.mean(mid_s_xy)}")

    # If the bug exists (minus sign in code), this will be -1.0
    assert np.allclose(
        mid_s_xy, 1.0, atol=1e-3
    ), f"Shear stress sign mismatch! Expected +1.0, got {np.mean(mid_s_xy)}"


def test_normal_stress_xx():
    """
    Test sigma_xx = d^2(phi)/dy_phy^2

    Coordinate transform:
        dy_phy = -dy_img
        d^2/dy_phy^2 = (-d/dy_img)^2 = d^2/dy_img^2

    Target: phi = y^2 (image coords)
    d^2(phi)/dy^2 = 2

    Expected: sigma_xx = 2
    """
    H, W = 50, 50
    bs = BSplineAiry((H, W), knot_spacing=5, degree=3)

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    phi = y**2

    C = fit_coefficients(bs, phi)

    s_xx, s_yy, s_xy = bs.get_stress_fields(C.flatten())

    margin = 10
    mid_s_xx = s_xx[margin:-margin, margin:-margin]

    assert np.allclose(mid_s_xx, 2.0, atol=1e-3), f"Sigma_xx mismatch! Expected 2.0, got {np.mean(mid_s_xx)}"


def test_normal_stress_yy():
    """
    Test sigma_yy = d^2(phi)/dx_phy^2

    Coordinate transform:
        dx_phy = dx_img

    Target: phi = x^2
    d^2(phi)/dx^2 = 2

    Expected: sigma_yy = 2
    """
    H, W = 50, 50
    bs = BSplineAiry((H, W), knot_spacing=5, degree=3)

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    phi = x**2

    C = fit_coefficients(bs, phi)

    s_xx, s_yy, s_xy = bs.get_stress_fields(C.flatten())

    margin = 10
    mid_s_yy = s_yy[margin:-margin, margin:-margin]

    assert np.allclose(mid_s_yy, 2.0, atol=1e-3), f"Sigma_yy mismatch! Expected 2.0, got {np.mean(mid_s_yy)}"
