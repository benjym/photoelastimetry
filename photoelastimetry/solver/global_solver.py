"""
Global stress measurement using Airy stress function.

This module implements a global inversion approach that solves for an Airy stress
function across the entire domain simultaneously. Unlike the local pixel-by-pixel
method, this approach:

1. Ensures mechanical equilibrium by construction (through Airy stress function)
2. Enforces smoothness globally via regularisation
3. Avoids local minima by solving a single global optimisation problem
4. Provides more stable results by incorporating spatial coupling

The Airy stress function φ(x,y) relates to stresses via:
    σ_xx = ∂²φ/∂y²
    σ_yy = ∂²φ/∂x²
    σ_xy = -∂²φ/∂x∂y
"""

import numpy as np
import scipy.optimize

from photoelastimetry.bspline import BSplineAiry
from photoelastimetry.image import (
    compute_principal_angle,
    compute_retardance,
    mueller_matrix,
    mueller_matrix_sensitivity,
)


def recover_stress_global(
    image_stack,
    wavelengths,
    C_values,
    nu,
    L,
    S_i_hat,
    knot_spacing=20,
    spline_degree=3,
    boundary_mask=None,
    regularization_weight=0.0,
    boundary_weight=1.0,
    max_iterations=100,
    tolerance=1e-5,
    analyzer_angles=None,
    verbose=True,
    debug=True,
    **kwargs,
):
    """
    Recover stress field by globally optimizing an Airy stress function B-spline surface.

    Parameters
    ----------
    image_stack : ndarray
        Shape (H, W, n_wavelengths, n_angles).
    wavelengths : list
        List of wavelengths.
    C_values : list
        List of stress-optic coefficients.
    nu : float or ndarray
        Solid fraction (usually 1.0).
    L : float
        Thickness.
    S_i_hat : list
        Incoming Stokes vector.
    knot_spacing : int
        Spacing of B-spline knots in pixels.
    boundary_mask : ndarray (bool), optional
        Mask where stress should be enforced to boundary_value.
    boundary_weight : float
        Weight for boundary condition penalty.
    """

    H, W, n_wl, n_ang = image_stack.shape

    # Preconditioner / Scale factor
    # Airy coefficients result in stresses. Stress = d2Phi.
    # d2B/dx2 scales as 1/spacing^2.
    # We estimate max stress corresponding to ~6 fringes (user estimate).
    # N = C * L * (s1-s2) / lambda  =>  stress ~ N * lambda / (C * L)

    avg_wl = np.mean(wavelengths)
    avg_C = np.mean(C_values)

    # Estimate based on ~6 fringes
    EST_FRINGES = 6.0
    estimated_stress = EST_FRINGES * avg_wl / (avg_C * L)

    # Second derivative of B-spline scales with 1/knot_spacing^2
    # So we need coefficients to be stress * knot_spacing^2
    COEFF_SCALE = estimated_stress * (knot_spacing**2)

    if verbose:
        print(f"Global Solver: Estimated max stress {estimated_stress:.2e} Pa (6 fringes).")
        print(f"Global Solver: Using coefficient scale factor {COEFF_SCALE:.2e}")

    if analyzer_angles is None:
        analyzer_angles = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])

    # Identify NaN regions in input data for logging, but do NOT automatically treat them as boundaries
    nan_mask = np.any(np.isnan(image_stack), axis=(2, 3))

    if verbose and np.any(nan_mask):
        print(f"Global Solver: Found {np.sum(nan_mask)} pixels with NaN data (will be ignored in fitting).")

    # Initialize B-Spline basis
    bspline = BSplineAiry((H, W), knot_spacing=knot_spacing, degree=spline_degree)

    if verbose:
        print(
            f"Global Solver: Image {H}x{W}, Knots {knot_spacing}px, "
            f"Coeffs {bspline.n_coeffs_y}x{bspline.n_coeffs_x} ({bspline.n_coeffs})"
        )

    # Pre-calculate constants for forward model
    # Convert lists to arrays for speed
    wavelengths = np.array(wavelengths)
    C_values = np.array(C_values)

    # Pre-compute trigonometric values for analyzer
    # Intensity = 0.5 * (S0 + S1 cos(2a) + S2 sin(2a))
    # S_out = M @ S_in
    # S_in = I0 * [1, S1_hat, S2_hat, S3_hat]

    S_in = np.array([1.0, S_i_hat[0], S_i_hat[1], S_i_hat[2]])

    cos_2a = np.cos(2 * analyzer_angles)
    sin_2a = np.sin(2 * analyzer_angles)

    # Helper to calculate predicted intensity for all pixels at once
    def forward_model(s_xx, s_yy, s_xy):
        # s_xx shape: (H, W). Returns (H, W, n_wl, n_ang)

        # Calculate derived properties
        # Orientation is independent of wavelength
        theta = compute_principal_angle(s_xx, s_yy, s_xy)  # (H, W)

        preds = []

        for i, (wl, C) in enumerate(zip(wavelengths, C_values)):
            delta = compute_retardance(s_xx, s_yy, s_xy, C, nu, L, wl)  # (H, W)

            # Mueller matrix per pixel: M is (H, W, 4, 4)
            M = mueller_matrix(theta, delta)

            # Transmitted Stokes vector
            # S_out = M @ S_in
            # S_in is (4,). We need broadcast.
            # M: (H,W,4,4), S_in: (4,) -> sum over axis 3
            # We want S_out to be (4, H, W)
            S_out = np.einsum("...ij,j->i...", M, S_in)  # (4, H, W)

            # Calibration / Analyser
            # I = 0.5 * (S0 + S1 cos + S2 sin)

            I_wl = np.zeros((H, W, n_ang))
            for a in range(n_ang):
                I_wl[..., a] = 0.5 * (S_out[0] + S_out[1] * cos_2a[a] + S_out[2] * sin_2a[a])

            preds.append(I_wl)

        # Stack to (H, W, n_wl, n_ang)
        return np.stack(preds, axis=2)

    # Create mask for valid pixels (non-NaN data)
    valid_mask = ~np.isnan(image_stack)
    if not np.all(valid_mask) and verbose:
        print(f"Global Solver: Ignoring {np.sum(~valid_mask)} NaN values in input image.")

    # Optimization counter for verbose output
    iteration_count = [0]

    # Objective function
    def loss_and_gradient(coeffs_flat):
        iteration_count[0] += 1

        # 1. Get stresses
        # Apply scaling to keep optimization variables ~1
        s_xx, s_yy, s_xy = bspline.get_stress_fields(coeffs_flat * COEFF_SCALE)

        # Initialize accumulated stress gradients
        grad_s_xx = np.zeros((H, W))
        grad_s_yy = np.zeros((H, W))
        grad_s_xy = np.zeros((H, W))
        loss = 0.0

        # Pre-calc geometry / stress invariants
        diff = s_xx - s_yy
        denom_sq = diff**2 + 4 * s_xy**2

        # Safe math for derivatives (avoid singularity at 0 stress)
        safe_mask = denom_sq > 1e-20
        safe_denom_sq = np.ones_like(denom_sq)
        safe_denom_sq[safe_mask] = denom_sq[safe_mask]

        sqrt_D = np.sqrt(safe_denom_sq)
        inv_denom = 1.0 / safe_denom_sq
        inv_sqrt = 1.0 / sqrt_D

        # Masked derivatives of Theta and Sqrt(D) w.r.t (diff, 2xy)
        # theta = 0.5 * atan2(2xy, diff)
        # dTheta/d(diff) = -0.5 * 2xy / D_sq = -xy/D_sq
        # dTheta/d(2xy) = 0.5 * diff / D_sq

        # D_sqrt = sqrt(diff^2 + (2xy)^2)
        # dSqrt/d(diff) = diff / sqrt
        # dSqrt/d(2xy) = 2xy / sqrt

        dTheta_ddiff = -s_xy * inv_denom
        dTheta_d2xy = 0.5 * diff * inv_denom

        dSqrt_ddiff = diff * inv_sqrt
        dSqrt_d2xy = 2 * s_xy * inv_sqrt

        # Zero out gradients in singular regions
        dTheta_ddiff[~safe_mask] = 0
        dTheta_d2xy[~safe_mask] = 0
        dSqrt_ddiff[~safe_mask] = 0
        dSqrt_d2xy[~safe_mask] = 0

        # Compute Theta (orientation) - shared across wavelengths
        # Use simple atan2, it handles 0,0 correctly (returns 0)
        theta = 0.5 * np.arctan2(2 * s_xy, diff)

        # 2. Iterate wavelengths to build Loss and Gradients
        for i, (wl, C) in enumerate(zip(wavelengths, C_values)):
            # Retardance
            K = 2 * np.pi * C * nu * L / wl
            delta = K * sqrt_D  # Mask not needed for value, just gradient stability
            if not np.all(safe_mask):
                delta[~safe_mask] = 0  # Correct limit

            # Forward Model
            M = mueller_matrix(theta, delta)  # (H, W, 4, 4)
            dM_dtheta, dM_ddelta = mueller_matrix_sensitivity(theta, delta)

            # Incident Light S_in (4,) broadcast
            # S_out = M @ S_in
            S_out = np.einsum("...ij,j->...i", M, S_in)  # (H, W, 4)

            # Sensitivity of S_out
            dS_dTheta = np.einsum("...ij,j->...i", dM_dtheta, S_in)
            dS_dDelta = np.einsum("...ij,j->...i", dM_ddelta, S_in)

            # Analysers loop
            for a in range(n_ang):
                # I_pred = 0.5 * (S0 + S1 c + S2 s)
                term = S_out[..., 0] + S_out[..., 1] * cos_2a[a] + S_out[..., 2] * sin_2a[a]
                I_pred = 0.5 * term

                # Residuals
                obs = image_stack[:, :, i, a]
                valid = ~np.isnan(obs)
                resid = np.zeros_like(I_pred)
                resid[valid] = I_pred[valid] - obs[valid]

                loss += np.sum(resid**2)

                # Gradients w.r.t I_pred
                dL_dI = 2 * resid

                # Partial derivatives of Intensity w.r.t Theta and Delta
                dI_dTheta = 0.5 * (
                    dS_dTheta[..., 0] + dS_dTheta[..., 1] * cos_2a[a] + dS_dTheta[..., 2] * sin_2a[a]
                )
                dI_dDelta = 0.5 * (
                    dS_dDelta[..., 0] + dS_dDelta[..., 1] * cos_2a[a] + dS_dDelta[..., 2] * sin_2a[a]
                )

                # Chain rule back to Theta/Delta
                dL_dTheta = dL_dI * dI_dTheta
                dL_dDelta = dL_dI * dI_dDelta

                # Chain rule back to Stress Components
                # dL/ds_xx = dL/dTheta * dTheta/d_xx + dL/dDelta * dDelta/d_xx
                # d_xx -> diff (+1)
                # d_yy -> diff (-1)
                # d_xy -> 2xy (2)

                # Precompute shared terms
                term_xx = dL_dTheta * dTheta_ddiff + dL_dDelta * (K * dSqrt_ddiff)
                term_xy = dL_dTheta * dTheta_d2xy + dL_dDelta * (K * dSqrt_d2xy)

                grad_s_xx += term_xx
                grad_s_yy -= term_xx  # (diff depends on -s_yy)
                grad_s_xy += term_xy * 2  # (2xy depends on 2*s_xy)

        # 4. Boundary penalty
        if boundary_mask is not None:
            stress_mag = s_xx**2 + s_yy**2 + 2 * s_xy**2
            bc_loss = np.sum(stress_mag[boundary_mask])
            loss += boundary_weight * bc_loss

            # Gradient
            mask_w = boundary_mask.astype(float) * boundary_weight * 2
            grad_s_xx += mask_w * s_xx
            grad_s_yy += mask_w * s_yy
            grad_s_xy += mask_w * 2 * s_xy  # from 2*s_xy^2 -> 4 s_xy

        # 5. Project Stress Gradients to Coeffs
        grad_coeffs = bspline.project_stress_gradients(grad_s_xx, grad_s_yy, grad_s_xy)
        grad_coeffs *= COEFF_SCALE

        # 6. Regularization (Smoothness)
        if regularization_weight > 0:
            C_grid = coeffs_flat.reshape(bspline.n_coeffs_y, bspline.n_coeffs_x)
            diff_y = np.diff(C_grid, axis=0)
            diff_x = np.diff(C_grid, axis=1)

            reg_term = np.sum(diff_y**2) + np.sum(diff_x**2)
            loss += regularization_weight * reg_term

            # Gradient of regularization
            grad_reg_y = np.zeros_like(C_grid)
            grad_reg_y[:-1, :] -= 2 * diff_y
            grad_reg_y[1:, :] += 2 * diff_y

            grad_reg_x = np.zeros_like(C_grid)
            grad_reg_x[:, :-1] -= 2 * diff_x
            grad_reg_x[:, 1:] += 2 * diff_x

            grad_coeffs += regularization_weight * (grad_reg_y + grad_reg_x).flatten()

        if verbose and iteration_count[0] % 10 == 0:
            print(f"Iteration {iteration_count[0]}, Loss: {loss:.6e}", end="\r")

        if debug:
            # plot all intermediate results occaisionally
            # if iteration_count[0] % 50 == 1:
            import matplotlib.pyplot as plt

            print(f"\nDebug Plot at iteration {iteration_count[0]}, Loss: {loss:.6e}")
            plt.figure(0, figsize=(12, 4))
            plt.clf()
            for i, (s_field, label) in enumerate(
                zip([s_xx, s_yy, s_xy], ["Sigma_xx", "Sigma_yy", "Sigma_xy"])
            ):
                plt.subplot(1, 3, i + 1)
                plt.imshow(s_field, cmap="viridis")
                plt.colorbar()
                plt.title(label)
            plt.suptitle(f"Iteration {iteration_count[0]}, Loss: {loss:.6e}")
            plt.tight_layout()
            plt.pause(0.1)

        return loss, grad_coeffs

    # Initialize with small random noise to avoid singularity at 0 stress
    # The gradient of sqrt(stress^2) is undefined at 0, which can cause L-BFGS-B to fail
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    # Start with noise ~0.1% of the expected stress range to ensure we are outside the singular region
    # (Singularity protection threshold is stress^2 < 1e-20)
    initial_coeffs = rng.normal(0, 1e-10, bspline.n_coeffs)
    # initial_coeffs = np.zeros(bspline.n_coeffs)

    # Use L-BFGS-B
    res = scipy.optimize.minimize(
        loss_and_gradient,
        initial_coeffs,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": tolerance, "disp": verbose},
    )

    if verbose:
        print(f"Optimization finished: {res.message}")

    final_s_xx, final_s_yy, final_s_xy = bspline.get_stress_fields(res.x * COEFF_SCALE)

    # wipe out anywhere input was NaN
    final_s_xx[nan_mask] = np.nan
    final_s_yy[nan_mask] = np.nan
    final_s_xy[nan_mask] = np.nan

    # Return (H, W, 3) like stress_map
    return np.stack([final_s_xx, final_s_yy, final_s_xy], axis=-1)
