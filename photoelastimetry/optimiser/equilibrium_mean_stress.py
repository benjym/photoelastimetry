"""
Global Mean Stress Recovery Solver.

This module provides a solver that reconstructs the hydrostatic (mean) stress component
given the known deviatoric stress components (derived from photoelastic measurements),
while enforcing global equilibrium using the Airy stress function.

Unlike the full equilibrium solver, this approach assumes the principal stress difference
and orientation are "trusted" inputs (e.g. from high-quality seeding), and only optimizes
the stress field to be consistent with these inputs while satisfying equilibrium.
"""

import numpy as np
import scipy.optimize

from photoelastimetry.bspline import BSplineAiry


def recover_mean_stress(
    delta_sigma_map,
    theta_map,
    knot_spacing=20,
    spline_degree=3,
    boundary_mask=None,
    boundary_values=None,
    boundary_weight=1.0,
    regularisation_weight=0.0,
    regularisation_order=2,
    external_potential=None,
    max_iterations=100,
    tolerance=1e-5,
    initial_stress_map=None,
    verbose=True,
    debug=False,
    **kwargs,
):
    """
    Recover the full stress field by finding an Airy stress function that best matches
    the observed deviatoric stress components.

    Parameters
    ----------
    delta_sigma_map : ndarray
        Map of principal stress difference [H, W] (Pa).
    theta_map : ndarray
        Map of principal stress orientation [H, W] (radians).
    knot_spacing : int
        Spacing of B-spline knots in pixels.
    boundary_mask : ndarray (bool), optional
        Mask where specific stress component values should be enforced.
    boundary_values : dict, optional
        Dictionary containing 'xx', 'yy', 'xy' keys with target boundary values maps (Pa).
        Only pixels where boundary_mask is True are used.
    boundary_weight : float
        Weight for boundary condition penalty.
    regularisation_weight : float
        Weight for smoothness regularisation on B-spline coefficients.
    external_potential : ndarray, optional
        Scalar field V(x,y) [H, W] representing body force potential (e.g. -rho*g*y).
        This field is added to sigma_xx and sigma_yy:
        sigma_xx_total = d2phi/dy2 + V
        sigma_yy_total = d2phi/dx2 + V
        If provided, the solver optimizes phi such that total stresses match observations.
    initial_stress_map : ndarray, optional
        Initial guess for stress field [H, W, 3] to seed B-spline.
    boundary_weight : float
        Weight for boundary condition penalty.
    initial_stress_map : ndarray, optional
        Initial guess for stress field [H, W, 3] to seed B-spline.

    Returns
    -------
    bspline : BSplineAiry
        Fitted B-spline object.
    """

    H, W = delta_sigma_map.shape

    # 1. Compute target deviatoric components from input maps
    # We want the solver to find stresses such that:
    #   sigma_xx - sigma_yy  approx  delta_sigma * cos(2*theta)
    #   sigma_xy             approx  0.5 * delta_sigma * sin(2*theta)

    target_diff = delta_sigma_map * np.cos(2 * theta_map)
    target_shear = 0.5 * delta_sigma_map * np.sin(2 * theta_map)

    # Scale factor estimation
    # Use max stress for scaling coefficients
    max_stress = np.nanmax(delta_sigma_map)
    if max_stress == 0 or np.isnan(max_stress):
        max_stress = 1.0

    if verbose:
        print(f"Mean Stress Solver: Max stress {max_stress:.2e} Pa.")

    # Handle NaNs in input
    valid_mask = ~(np.isnan(target_diff) | np.isnan(target_shear))
    if verbose and not np.all(valid_mask):
        print(f"Mean Stress Solver: Ignoring {np.sum(~valid_mask)} pixels with NaN inputs.")

    target_diff[~valid_mask] = 0
    target_shear[~valid_mask] = 0

    # Initialize B-Spline basis
    bspline = BSplineAiry((H, W), knot_spacing=knot_spacing, degree=spline_degree)

    if verbose:
        print(
            f"Mean Stress Solver: Grid {H}x{W}, Knots {knot_spacing}px, "
            f"Coeffs {bspline.n_coeffs_y}x{bspline.n_coeffs_x} ({bspline.n_coeffs})"
        )

    # Optimization counter
    iteration_count = [0]

    def loss_and_gradient(coeffs_flat):
        iteration_count[0] += 1

        # Calculate stresses from current Airy function
        s_xx, s_yy, s_xy = bspline.get_stress_fields(coeffs_flat)
        # Add external potential (body forces)
        if external_potential is not None:
            s_xx = s_xx + external_potential
            s_yy = s_yy + external_potential
        # Derived deviatoric components
        model_diff = s_xx - s_yy
        model_shear = s_xy

        # Residuals
        resid_diff = model_diff - target_diff
        resid_shear = model_shear - target_shear

        # Mask invalid pixels
        resid_diff[~valid_mask] = 0
        resid_shear[~valid_mask] = 0

        # Loss function
        # L = sum( (s_xx - s_yy - T_diff)^2 + 4 * (s_xy - T_shear)^2 )
        # Factor of 4 on shear is optional but makes it consistent with L2 norm of Deviatoric vector
        # (diff, 2*shear). Or we can treat them equally. Let's treat them equally (factor 1).
        # Actually, let's stick to simple least squares on components.
        loss = np.sum(resid_diff**2) + 4 * np.sum(resid_shear**2)

        # Gradients w.r.t stresses
        # dL/d(s_xx) = 2 * resid_diff
        # dL/d(s_yy) = -2 * resid_diff
        # dL/d(s_xy) = 8 * resid_shear

        grad_s_xx = 2 * resid_diff
        grad_s_yy = -2 * resid_diff
        grad_s_xy = 8 * resid_shear

        # Boundary penalty
        if boundary_mask is not None:
            # If boundary_values are provided, penalise deviation from them
            # If not provided, assume zero stress at boundary (for now, or raise error)

            mask_w = boundary_mask.astype(float) * boundary_weight

            if boundary_values is not None:
                # Targeted boundary conditions
                # Only enforce components present in dictionary

                if "xx" in boundary_values:
                    b_xx = boundary_values["xx"]
                    res_b_xx = s_xx - b_xx
                    # Check for NaNs in boundary values to allow per-pixel freedom
                    if np.isnan(b_xx).any():
                        valid_b = ~np.isnan(b_xx)
                        # Only apply where both boundary_mask is true AND value is valid
                        active_mask = boundary_mask & valid_b
                        loss += boundary_weight * np.sum(res_b_xx[active_mask] ** 2)
                        grad_s_xx[active_mask] += 2 * boundary_weight * res_b_xx[active_mask]
                    else:
                        loss += boundary_weight * np.sum(res_b_xx[boundary_mask] ** 2)
                        grad_s_xx += 2 * mask_w * res_b_xx

                if "yy" in boundary_values:
                    b_yy = boundary_values["yy"]
                    res_b_yy = s_yy - b_yy
                    if np.isnan(b_yy).any():
                        valid_b = ~np.isnan(b_yy)
                        active_mask = boundary_mask & valid_b
                        loss += boundary_weight * np.sum(res_b_yy[active_mask] ** 2)
                        grad_s_yy[active_mask] += 2 * boundary_weight * res_b_yy[active_mask]
                    else:
                        loss += boundary_weight * np.sum(res_b_yy[boundary_mask] ** 2)
                        grad_s_yy += 2 * mask_w * res_b_yy

                if "xy" in boundary_values:
                    b_xy = boundary_values["xy"]
                    res_b_xy = s_xy - b_xy
                    if np.isnan(b_xy).any():
                        valid_b = ~np.isnan(b_xy)
                        active_mask = boundary_mask & valid_b
                        loss += boundary_weight * np.sum(res_b_xy[active_mask] ** 2)
                        grad_s_xy[active_mask] += 2 * boundary_weight * res_b_xy[active_mask]
                    else:
                        loss += boundary_weight * np.sum(res_b_xy[boundary_mask] ** 2)
                        grad_s_xy += 2 * mask_w * res_b_xy

            else:
                # Default: zero stress magnitude (maybe aggressive?)
                # Assuming free boundary -> zero normal/shear?
                # Let's enforce zero stress vector magnitude if not specified?
                # Or user should pass boundary values.
                # Reverting to "zero stress magnitude" behavior of original solver if no values
                stress_sq = s_xx**2 + s_yy**2 + 2 * s_xy**2
                loss += boundary_weight * np.sum(stress_sq[boundary_mask])

                # Gradient
                grad_s_xx += 2 * mask_w * s_xx
                grad_s_yy += 2 * mask_w * s_yy
                grad_s_xy += 4 * mask_w * s_xy

        # regularisation (Smoothness of coefficients)
        grad_coeffs = bspline.project_stress_gradients(grad_s_xx, grad_s_yy, grad_s_xy)

        if regularisation_weight > 0:
            C_grid = coeffs_flat.reshape(bspline.n_coeffs_y, bspline.n_coeffs_x)

            grad_reg_y = np.zeros_like(C_grid)
            grad_reg_x = np.zeros_like(C_grid)
            reg_term = 0.0

            if regularisation_order == 1:
                # 1st order difference (Gradient penalty)
                diff_y = np.diff(C_grid, axis=0)
                diff_x = np.diff(C_grid, axis=1)

                reg_term = np.sum(diff_y**2) + np.sum(diff_x**2)
                loss += regularisation_weight * reg_term

                grad_reg_y[:-1, :] -= 2 * diff_y
                grad_reg_y[1:, :] += 2 * diff_y
                grad_reg_x[:, :-1] -= 2 * diff_x
                grad_reg_x[:, 1:] += 2 * diff_x

            elif regularisation_order == 2:
                # 2nd order difference (Curvature penalty)
                if C_grid.shape[0] > 2:
                    diff2_y = np.diff(C_grid, n=2, axis=0)
                    reg_term += np.sum(diff2_y**2)

                    term_y = 2 * diff2_y
                    grad_reg_y[:-2, :] += term_y
                    grad_reg_y[1:-1, :] -= 2 * term_y
                    grad_reg_y[2:, :] += term_y

                if C_grid.shape[1] > 2:
                    diff2_x = np.diff(C_grid, n=2, axis=1)
                    reg_term += np.sum(diff2_x**2)

                    term_x = 2 * diff2_x
                    grad_reg_x[:, :-2] += term_x
                    grad_reg_x[:, 1:-1] -= 2 * term_x
                    grad_reg_x[:, 2:] += term_x

                loss += regularisation_weight * reg_term

            grad_coeffs += regularisation_weight * (grad_reg_y + grad_reg_x).flatten()

        if verbose and iteration_count[0] % 10 == 0:
            print(f"Mean Stress Solver Iteration {iteration_count[0]}, Loss: {loss:.6e}", end="\r")

        if debug and iteration_count[0] % 50 == 1:
            import matplotlib.pyplot as plt

            # minimal debug plotting (optional)
            pass

        return loss, grad_coeffs

    # Initialize coefficients
    # Should probably initialize from measurement if possible
    # We can fit a spline to the deviatoric measurements?
    # Hard to fit p if it's unknown.
    # Start from zero or initial_stress_map if provided.

    if initial_stress_map is not None:
        if verbose:
            print("Mean Stress Solver: Initializing from supplied stress map...")
        fitted_coeffs = bspline.fit_stress_field(initial_stress_map)
        initial_coeffs = fitted_coeffs
    else:
        rng = np.random.default_rng(42)
        initial_coeffs = rng.normal(0, 1e-10, bspline.n_coeffs)

    # Optimization
    res = scipy.optimize.minimize(
        loss_and_gradient,
        initial_coeffs,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": tolerance, "disp": verbose},
    )

    if verbose:
        print(f"\nOptimization finished: {res.message}")

    return bspline, res.x
