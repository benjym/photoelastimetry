"""
Global Mean Stress Recovery Solver.

This module provides a solver that reconstructs the hydrostatic (mean) stress component
given the known deviatoric stress components (derived from photoelastic measurements),
while enforcing global equilibrium in a least-squares sense using a scalar B-spline
pressure field.

Unlike the full equilibrium solver, this approach assumes the principal stress difference
and orientation are "trusted" inputs (e.g. from high-quality seeding), and only optimizes
the stress field to be consistent with these inputs while satisfying equilibrium.
"""

import numpy as np
import scipy.optimize

from photoelastimetry.bspline import BSplineScalar


class PressureFieldResult:
    """Wrapper to present Pressure Field results as full stress fields."""

    def __init__(self, bspline_p, s_xx_dev, s_yy_dev, s_xy_dev):
        self.bspline_p = bspline_p
        self.s_xx_dev = s_xx_dev
        self.s_yy_dev = s_yy_dev
        self.s_xy_dev = s_xy_dev
        # Forward basic properties
        self.n_coeffs = bspline_p.n_coeffs
        self.n_coeffs_x = bspline_p.n_coeffs_x
        self.n_coeffs_y = bspline_p.n_coeffs_y

    def get_stress_fields(self, coeffs):
        """
        Reconstruct total stress fields from pressure coefficients and stored deviatoric parts.

        Sigma_tot = P + S_dev
        Note: If external potential V is used, it should be added externally to XX and YY.
        """
        # Calculate P
        P, _, _ = self.bspline_p.get_scalar_fields(coeffs)

        # Combine
        # s_xx_dev, s_yy_dev, s_xy_dev MUST be broadcastable or same shape
        # They are stored as full maps.

        sigma_xx = P + self.s_xx_dev
        sigma_yy = P + self.s_yy_dev
        sigma_xy = self.s_xy_dev

        return sigma_xx, sigma_yy, sigma_xy

    def project_stress_gradients(self, grad_s_xx, grad_s_yy, grad_s_xy):
        """Projections for regularization (if needed externally)."""
        # Gradient of Loss w.r.t Stresses -> Coefficients
        # Stress = P + S_dev
        # dStress/dP = 1
        # dLoss/dP = dLoss/dStress * 1

        grad_P = grad_s_xx + grad_s_yy  # + 0 * grad_s_xy

        # P = B * C
        return self.bspline_p.project_scalar_gradients(grad_P, None, None)


def stress_to_principal_invariants(stress_map):
    """
    Convert a stress map [sigma_xx, sigma_yy, sigma_xy] to (delta_sigma, theta).

    Parameters
    ----------
    stress_map : ndarray
        Stress tensor map with shape (..., 3) ordered as [xx, yy, xy].

    Returns
    -------
    delta_sigma_map : ndarray
        Principal stress difference sqrt((xx-yy)^2 + 4*xy^2).
    theta_map : ndarray
        Principal stress angle 0.5*atan2(2*xy, xx-yy).
    """
    s_xx = stress_map[..., 0]
    s_yy = stress_map[..., 1]
    s_xy = stress_map[..., 2]

    delta_sigma_map = np.sqrt((s_xx - s_yy) ** 2 + 4 * s_xy**2)
    theta_map = 0.5 * np.arctan2(2 * s_xy, s_xx - s_yy)
    return delta_sigma_map, theta_map


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
    Recover the hydrostatic (mean) stress component P(x,y) given fixed deviatoric components.

    This solver assumes the deviatoric stress field (difference and orientation) provided
    as input is trusted and fixed. It solves for the scalar pressure field P(x,y) that
    best satisfies the stress equilibrium equations in a least-squares sense using a B-Spline
    representation.

    Stresses are constructed as:
        sigma_xx = P(x,y) + s_xx_meas
        sigma_yy = P(x,y) + s_yy_meas
        sigma_xy = s_xy_meas

    Equilibrium requires:
        d(sigma_xx)/dx + d(sigma_xy)/dy = 0  =>  dP/dx = -d(s_xx_meas)/dx - d(s_xy_meas)/dy
        d(sigma_xy)/dx + d(sigma_yy)/dy = 0  =>  dP/dy = -d(s_xy_meas)/dx - d(s_yy_meas)/dy

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
        This term is added to the equilibrium equations:
            d(sigma_xx)/dx + ... + F_x = 0  (where F_x = -dV/dx)
        If external_potential represents V where F = -grad(V), then:
            dP/dx = ... + dV/dx
            dP/dy = ... + dV/dy
        Effectively, P_total = P_solved - V.
        However, the current implementation treats input V as adding to normal stresses directly?
        Let's assume standard gravity potential: sigma_ij_total = sigma_ij_effective + delta_ij * V?
        Or simply Body Forces F_i.
        If external_potential is provided as V (potential) such that F = -grad(V).
    initial_stress_map : ndarray, optional
        Initial guess for stress field to seed P.

    Returns
    -------
    bspline : BSplineScalar
        Fitted B-spline object for Pressure field.
    coeffs : ndarray
        Coefficients for P.
    """

    H, W = delta_sigma_map.shape

    if external_potential is not None:
        external_potential = np.asarray(external_potential)

    if external_potential is not None and external_potential.shape != (H, W):
        raise ValueError(f"external_potential must have shape {(H, W)}, got {external_potential.shape}")

    # 1. Compute trusted deviatoric components
    # s_xx_dev = 0.5 * delta * cos(2theta)
    # s_yy_dev = -0.5 * delta * cos(2theta)
    # s_xy_dev = 0.5 * delta * sin(2theta)

    cos_2t = np.cos(2 * theta_map)
    sin_2t = np.sin(2 * theta_map)

    s_xx_meas = 0.5 * delta_sigma_map * cos_2t
    s_yy_meas = -0.5 * delta_sigma_map * cos_2t
    s_xy_meas = 0.5 * delta_sigma_map * sin_2t

    # Handle NaNs in input
    valid_mask = ~(np.isnan(delta_sigma_map) | np.isnan(theta_map))
    if external_potential is not None:
        valid_mask &= ~np.isnan(external_potential)

    if verbose and not np.all(valid_mask):
        print(f"Mean Stress Solver: Ignoring {np.sum(~valid_mask)} pixels with NaN inputs.")
        # Fill NaNs with 0 (gradients will be messy at edges of NaNs, but masking handles loss)
        s_xx_meas[~valid_mask] = 0
        s_yy_meas[~valid_mask] = 0
        s_xy_meas[~valid_mask] = 0

    # 2. Compute target gradients for P
    # Gradients of measured fields using central difference
    # gradient returns list [d/dy, d/dx]
    diff_s_xx = np.gradient(s_xx_meas)
    ds_xx_dy, ds_xx_dx = diff_s_xx

    diff_s_yy = np.gradient(s_yy_meas)
    ds_yy_dy, ds_yy_dx = diff_s_yy

    diff_s_xy = np.gradient(s_xy_meas)
    ds_xy_dy, ds_xy_dx = diff_s_xy

    # Equilibrium Targets:
    # dP/dx = - (ds_xx/dx + ds_xy/dy) - F_x
    # dP/dy = - (ds_xy/dx + ds_yy/dy) - F_y

    # Body forces from external potential V
    # F = -grad(V). So term -F becomes +grad(V)
    # dP/dx_target = - (ds_xx/dx + ds_xy/dy) + dV/dx

    # dP/dx_target = - (ds_xx/dx + ds_xy/dy)
    # dP/dy_target = - (ds_xy/dx + ds_yy/dy)
    # (Body forces handled by implicit cancellation or external addition)

    target_grad_P_x = -(ds_xx_dx + ds_xy_dy)
    target_grad_P_y = -(ds_xy_dx + ds_yy_dy)

    if external_potential is not None:
        safe_V = np.nan_to_num(external_potential, nan=0.0)
        dV_dy, dV_dx = np.gradient(safe_V)
        target_grad_P_x += dV_dx
        target_grad_P_y += dV_dy

    if verbose:
        max_grad = np.max(np.sqrt(target_grad_P_x**2 + target_grad_P_y**2))
        print(f"Mean Stress Solver: Max target gradient {max_grad:.2e} Pa/px.")

    # Initialize B-Spline Scalara
    bspline_backing = BSplineScalar((H, W), knot_spacing=knot_spacing, degree=spline_degree)

    # Create Wrapper
    bspline_wrapper = PressureFieldResult(bspline_backing, s_xx_meas, s_yy_meas, s_xy_meas)

    has_pressure_bc_global = False
    if boundary_mask is not None and boundary_values is not None:
        if "xx" in boundary_values:
            has_pressure_bc_global |= np.any(boundary_mask & ~np.isnan(boundary_values["xx"]))
        if "yy" in boundary_values:
            has_pressure_bc_global |= np.any(boundary_mask & ~np.isnan(boundary_values["yy"]))

    if verbose:
        print(
            f"Mean Stress Solver: Grid {H}x{W}, Knots {knot_spacing}px, "
            f"Coeffs {bspline_backing.n_coeffs_y}x{bspline_backing.n_coeffs_x} ({bspline_backing.n_coeffs})"
        )

    # Optimization counter
    iteration_count = [0]

    def loss_and_gradient(coeffs_flat):
        iteration_count[0] += 1

        # Calculate P and gradients from current spline
        P, dP_dx, dP_dy = bspline_backing.get_scalar_fields(coeffs_flat)

        # Residuals in gradients (Equilibrium violation)
        res_grad_x = dP_dx - target_grad_P_x
        res_grad_y = dP_dy - target_grad_P_y

        # Mask invalid pixels (where input or gradients are bad)
        # Note: np.gradient at edges is less accurate.
        # We assume valid_mask covers the trust region.
        res_grad_x[~valid_mask] = 0
        res_grad_y[~valid_mask] = 0

        # Loss function
        loss = np.sum(res_grad_x**2 + res_grad_y**2)

        # Gradients w.r.t P fields
        # dL/d(dP/dx) = 2 * res_grad_x
        # dL/d(dP/dy) = 2 * res_grad_y
        # dL/dP = 0 (from equilibrium term)

        grad_dP_dx = 2 * res_grad_x
        grad_dP_dy = 2 * res_grad_y
        grad_P = np.zeros_like(P)

        # Boundary penalty on P values (Dirichlet conditions)
        # If we have known stresses at boundaries:
        # sigma_xx_bound = P + s_xx_meas (+ V if V used) => P_target = sigma_xx_bound - s_xx_meas (- V)

        if boundary_mask is not None and boundary_values is not None:
            # We enforce P to match implied pressure from boundary conditions
            # Prioritize Normal stresses which give P directly

            # Count how many conditions set P at each pixel
            count_P = np.zeros_like(P, dtype=float)
            sum_P_target = np.zeros_like(P)

            V_term = external_potential if external_potential is not None else 0

            if "xx" in boundary_values:
                b_xx = boundary_values["xx"]
                if not np.all(np.isnan(b_xx)):  # If useful
                    valid_b = ~np.isnan(b_xx) & boundary_mask
                    # P = sigma_xx - s_xx_dev - V
                    p_from_xx = b_xx - s_xx_meas - V_term
                    sum_P_target[valid_b] += p_from_xx[valid_b]
                    count_P[valid_b] += 1

            if "yy" in boundary_values:
                b_yy = boundary_values["yy"]
                if not np.all(np.isnan(b_yy)):
                    valid_b = ~np.isnan(b_yy) & boundary_mask
                    # P = sigma_yy - s_yy_dev - V
                    p_from_yy = b_yy - s_yy_meas - V_term
                    sum_P_target[valid_b] += p_from_yy[valid_b]
                    count_P[valid_b] += 1

            if "xy" in boundary_values:
                # XY boundary condition does not constrain P directly!
                # It constrains s_xy_meas, which is fixed input.
                pass

            # Average targets where consistent
            has_target = count_P > 0

            if np.any(has_target):
                P_target = np.zeros_like(P)
                P_target[has_target] = sum_P_target[has_target] / count_P[has_target]

                res_P = P - P_target

                loss += boundary_weight * np.sum(res_P[has_target] ** 2)
                grad_P[has_target] += 2 * boundary_weight * res_P[has_target]

        elif boundary_mask is not None:
            # Fallback
            pass

        # Gauge condition for under-constrained pressure: center mean pressure at zero.
        if not has_pressure_bc_global:
            n_valid = int(np.sum(valid_mask))
            if n_valid > 0:
                mean_P = np.mean(P[valid_mask])
                gauge_weight = 1.0
                loss += gauge_weight * mean_P**2
                grad_P[valid_mask] += (2 * gauge_weight * mean_P) / n_valid

        # Backproject gradients to coefficients
        grad_coeffs = bspline_backing.project_scalar_gradients(grad_P, grad_dP_dx, grad_dP_dy)

        # Smoothness regularization
        if regularisation_weight > 0:
            grad_coeffs += regularisation_weight * coeffs_flat  # Simple L2 Ridge
            loss += 0.5 * regularisation_weight * np.sum(coeffs_flat**2)

        if verbose and iteration_count[0] % 10 == 0:
            print(f"Mean Stress Solver Iteration {iteration_count[0]}, Loss: {loss:.6e}", end="\r")

        return loss, grad_coeffs

    # Initialize coefficients
    if initial_stress_map is not None:
        # Estimate P form initial map
        # initial_stress_map contains full [xx, yy, xy]
        s_xx_init = initial_stress_map[:, :, 0]
        s_yy_init = initial_stress_map[:, :, 1]

        # P_init = (s_xx + s_yy)/2 - V? No, P is just (s_xx + s_yy)/2 - s_dev...
        # s_xx = P + s_xx_dev => P = s_xx - s_xx_dev
        # s_yy = P + s_yy_dev => P = s_yy - s_yy_dev
        # Average: P = Mean_Stress - (s_xx_dev + s_yy_dev)/2
        # But s_xx_dev = -s_yy_dev, so (s_xx_dev + s_yy_dev)/2 = 0
        # So P_init = Mean_Stress = (s_xx + s_yy)/2

        # We must subtract V if initial map includes V
        V_init = external_potential if external_potential is not None else 0

        P_init_field = (s_xx_init + s_yy_init) / 2.0 - V_init

        # Fit B-spline coefficients to the initial pressure estimate for faster convergence.
        initial_coeffs = bspline_backing.fit_scalar_field(P_init_field, mask=valid_mask, maxiter=100)
    else:
        rng = np.random.default_rng(42)
        initial_coeffs = rng.normal(0, 1e-10, bspline_backing.n_coeffs)

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

    # If pressure BCs do not pin the gauge, shift coefficients to exactly zero-mean pressure.
    if not has_pressure_bc_global and np.any(valid_mask):
        P_opt, _, _ = bspline_backing.get_scalar_fields(res.x)
        mean_P = np.mean(P_opt[valid_mask])
        res.x = res.x - mean_P

    return bspline_wrapper, res.x
