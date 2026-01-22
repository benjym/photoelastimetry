"""
Local stress measurement using polarimetric imaging.

This module implements the local stress measurement algorithm using Mueller
matrix calculus and multi-wavelength polarimetry to recover the full 2D stress
tensor at each pixel from polarimetric images.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from tqdm import tqdm

from photoelastimetry.image import (
    compute_normalised_stokes,
    compute_principal_angle,
    compute_retardance,
    compute_stokes_components,
    mueller_matrix,
)


def predict_stokes(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength, S_i_hat):
    """
    Predict normalised Stokes vector components from stress tensor.

    Parameters
    ----------
    sigma_xx : float
        Normal stress component in x direction (Pa).
    sigma_yy : float
        Normal stress component in y direction (Pa).
    sigma_xy : float
        Shear stress component (Pa).
    C : float
        Stress-optic coefficient (1/Pa).
    nu : float
        Solid fraction (use 1.0 for solid samples).
    L : float
        Sample thickness (m).
    wavelength : float
        Wavelength of light (m).
    S_i_hat : array-like
        Incoming normalised Stokes vector [S1_hat, S2_hat] or [S1_hat, S2_hat, S3_hat].
        If 2 elements, S3_hat is assumed to be 0 (no circular polarisation).
        If 3 elements, S3_hat represents circular polarisation component.

    Returns
    -------
    S_p_hat : ndarray
        Predicted normalised Stokes components [S1_hat, S2_hat].
    """
    theta = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)
    delta = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength)

    M = mueller_matrix(theta, delta)

    # Extend S_i_hat to full Stokes vector
    S_i_hat = np.asarray(S_i_hat)
    if len(S_i_hat) == 2:
        # Backward compatibility: assume S3 = 0 (no circular polarisation)
        S_i_full = np.array([1.0, S_i_hat[0], S_i_hat[1], 0.0])
    elif len(S_i_hat) == 3:
        # Use provided circular component
        S_i_full = np.array([1.0, S_i_hat[0], S_i_hat[1], S_i_hat[2]])
    else:
        raise ValueError(f"S_i_hat must have 2 or 3 elements, got {len(S_i_hat)}")

    # Apply Mueller matrix
    S_m = M @ S_i_full

    # Return normalised components (excluding S0)
    S_p_hat = S_m[1:3]

    return S_p_hat


def compute_residual(stress_params, S_m_hat, wavelengths, C_values, nu, L, S_i_hat):
    """
    Compute residual between measured and predicted Stokes components.

    Parameters
    ----------
    stress_params : array-like
        Stress tensor components [sigma_xx, sigma_yy, sigma_xy].
    S_m_hat : ndarray
        Measured normalised Stokes components, shape (3, 2) for RGB channels.
    wavelengths : array-like
        Wavelengths for R, G, B channels (m).
    C_values : array-like
        Stress-optic coefficients for R, G, B channels (1/Pa).
    nu : float
        Solid fraction (use 1.0 for solid samples).
    L : float
        Sample thickness (m).
    S_i_hat : array-like
        Incoming normalised Stokes vector [S1_hat, S2_hat] or [S1_hat, S2_hat, S3_hat].

    Returns
    -------
    residual : float
        Sum of squared residuals across all colour channels.
    """
    sigma_xx, sigma_yy, sigma_xy = stress_params

    residual = 0.0
    for c in range(3):  # R, G, B
        S_p_hat = predict_stokes(
            sigma_xx,
            sigma_yy,
            sigma_xy,
            C_values[c],
            nu,
            L,
            wavelengths[c],
            S_i_hat,
        )
        diff = S_m_hat[c] - S_p_hat
        residual += np.sum(diff**2)

    return residual


def _optimize_stress_tensor(
    S_m_hat,
    wavelengths,
    C_values,
    nu,
    L,
    S_i_hat,
    initial_guess,
    max_fringes,
    callback=None,
    track_all_paths=False,
):
    """
    Core optimization routine for stress tensor recovery.

    This is an internal function that performs the actual optimization.
    Use recover_stress_tensor() or recover_stress_tensor_live() instead.

    Parameters
    ----------
    S_m_hat : ndarray
        Measured normalised Stokes components, shape (3, 2) for RGB channels.
    wavelengths : array-like
        Wavelengths for R, G, B channels (m).
    C_values : array-like
        Stress-optic coefficients for R, G, B channels (1/Pa).
    nu : float
        Solid fraction.
    L : float
        Sample thickness (m).
    S_i_hat : array-like
        Incoming normalised Stokes vector [S1_hat, S2_hat] or [S1_hat, S2_hat, S3_hat].
    initial_guess : array-like
        Initial guess for stress tensor [sigma_xx, sigma_yy, sigma_xy].
    max_fringes : float
        Maximum expected fringe order for setting bounds.
    callback : callable, optional
        Callback function called at each iteration with current parameters.
    track_all_paths : bool, optional
        If True, return information about all optimization paths explored.

    Returns
    -------
    result : OptimizeResult
        Scipy optimization result object.
    all_paths : list of dict, optional
        Only returned if track_all_paths=True. Each dict contains:
        - 'stress_params': array of stress tensors at each iteration
        - 'S_predicted': array of predicted Stokes parameters
        - 'residuals': array of residuals
        - 'start_point': initial guess for this path
        - 'is_best': whether this path led to the best solution
    """
    # Compute stress bounds based on maximum fringe order
    wavelength_min = np.min(wavelengths)
    C_max = np.max(C_values)
    delta_max = max_fringes * 2 * np.pi
    stress_diff_max = delta_max * wavelength_min / (C_max * nu * L * 2 * np.pi)
    sigma_bound = 2.0 * stress_diff_max
    bounds = [(-sigma_bound, sigma_bound), (-sigma_bound, sigma_bound), (-sigma_bound, sigma_bound)]

    # Storage for all optimization paths
    all_paths = [] if track_all_paths else None

    # Helper function to create a callback that records history for a specific path
    def make_path_callback():
        path_history = {"stress_params": [], "S_predicted": [], "residuals": []}

        def path_callback(xk):
            path_history["stress_params"].append(xk.copy())
            # Compute predicted S_i_hat for all channels
            S_pred_all = np.zeros((3, 2))
            for c in range(3):
                S_pred_all[c] = predict_stokes(
                    xk[0], xk[1], xk[2], C_values[c], nu, L, wavelengths[c], S_i_hat
                )
            path_history["S_predicted"].append(S_pred_all.copy())
            # Compute residual
            residual = compute_residual(xk, S_m_hat, wavelengths, C_values, nu, L, S_i_hat)
            path_history["residuals"].append(residual)
            # Also call the original callback if provided
            if callback is not None:
                callback(xk)

        return path_callback, path_history

    # Multi-start optimization strategy based on periodicity in principal stress space
    # First try the provided initial guess
    if track_all_paths:
        path_callback, path_history = make_path_callback()
    else:
        path_callback = callback

    result = minimize(
        compute_residual,
        initial_guess,
        args=(S_m_hat, wavelengths, C_values, nu, L, S_i_hat),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 500},
        callback=path_callback,
    )

    if track_all_paths:
        all_paths.append(
            {
                "stress_params": np.array(path_history["stress_params"]),
                "S_predicted": np.array(path_history["S_predicted"]),
                "residuals": np.array(path_history["residuals"]),
                "start_point": initial_guess.copy(),
                "final_point": result.x.copy(),
                "final_residual": result.fun,
                "is_best": True,  # Will be updated later
            }
        )

    best_result = result

    # If not converged well, use multi-start based on fringe periodicity in principal stress space
    # if not result.success or result.fun > 1e-6:
    # Stress difference corresponding to one fringe of retardance
    fringe_stress = wavelength_min / (C_max * nu * L)

    # Sample principal stress differences (these create the fringes)
    delta_sigma_samples = []
    for n_fringes in np.linspace(0.5, max_fringes, int(max_fringes * 2)):
        delta_sigma_samples.append(n_fringes * fringe_stress)

    # Sample principal angles (0 to π, due to symmetry)
    # theta_samples = np.linspace(0, np.pi, 3, endpoint=False)
    theta_samples = np.array([0])

    # Sample mean stress
    sigma_mean_samples = np.linspace(-sigma_bound / 2, sigma_bound / 2, 5)

    start_points = []
    for delta_sigma in delta_sigma_samples:
        for theta in theta_samples:
            for sigma_mean in sigma_mean_samples:
                # Convert from principal stress representation to Cartesian
                sigma_xx = sigma_mean + (delta_sigma / 2) * np.cos(2 * theta)
                sigma_yy = sigma_mean - (delta_sigma / 2) * np.cos(2 * theta)
                sigma_xy = (delta_sigma / 2) * np.sin(2 * theta)

                # Check if within bounds
                if (
                    abs(sigma_xx) <= sigma_bound
                    and abs(sigma_yy) <= sigma_bound
                    and abs(sigma_xy) <= sigma_bound
                ):
                    start_points.append([sigma_xx, sigma_yy, sigma_xy])

    # Limit number of start points to keep runtime reasonable
    max_starts = 10000
    if len(start_points) > max_starts:
        indices = np.linspace(0, len(start_points) - 1, max_starts, dtype=int)
        start_points = [start_points[i] for i in indices]

    # Try optimization from each start point
    for start in start_points:
        if track_all_paths:
            path_callback, path_history = make_path_callback()
        else:
            path_callback = None

        result_new = minimize(
            compute_residual,
            start,
            args=(S_m_hat, wavelengths, C_values, nu, L, S_i_hat),
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 200},
            callback=path_callback,
        )

        if track_all_paths and len(path_history["residuals"]) > 0:
            all_paths.append(
                {
                    "stress_params": np.array(path_history["stress_params"]),
                    "S_predicted": np.array(path_history["S_predicted"]),
                    "residuals": np.array(path_history["residuals"]),
                    "start_point": np.array(start),
                    "final_point": result_new.x.copy(),
                    "final_residual": result_new.fun,
                    "is_best": False,  # Will be updated later
                }
            )

        if result_new.fun < best_result.fun:
            best_result = result_new
            # If we found a very good solution, stop early
            if result_new.fun < 1e-10:
                break

    # Mark the best path (use residual as primary criterion since final_point matching can be ambiguous)
    if track_all_paths:
        best_residual = best_result.fun
        best_path_idx = None
        for i, path in enumerate(all_paths):
            if np.abs(path["final_residual"] - best_residual) < 1e-10:
                path["is_best"] = True
                if best_path_idx is None:
                    best_path_idx = i
            else:
                path["is_best"] = False

    if track_all_paths:
        return best_result, all_paths
    else:
        return best_result


def recover_stress_tensor(
    S_m_hat, wavelengths, C_values, nu, L, S_i_hat, initial_guess=None, track_history=False, max_fringes=6
):
    """
    Recover stress tensor components by minimizing residual.

    Parameters
    ----------
    S_m_hat : ndarray
        Measured normalised Stokes components, shape (3, 2) for RGB channels.
        Each row is [S1_hat, S2_hat] for a colour channel.
    wavelengths : array-like
        Wavelengths for R, G, B channels (m).
    C_values : array-like
        Stress-optic coefficients for R, G, B channels (1/Pa).
    nu : float
        Solid fraction (use 1.0 for solid samples).
    L : float
        Sample thickness (m).
    S_i_hat : array-like
        Incoming normalised Stokes vector [S1_hat, S2_hat] or [S1_hat, S2_hat, S3_hat].
    initial_guess : array-like, optional
        Initial guess for stress tensor [sigma_xx, sigma_yy, sigma_xy].
        Default is [1, 1, 1].
    track_history : bool, optional
        If True, track optimization history for debugging plots. Default is False.
    max_fringes : float, optional
        Maximum expected fringe order. Used to set bounds on stress components.
        Default is 6 fringes, which corresponds to ~1.7 MPa for typical materials.

    Returns
    -------
    stress_tensor : ndarray
        Recovered stress tensor components [sigma_xx, sigma_yy, sigma_xy].
    success : bool
        Whether optimization was successful.
    history : dict, optional
        Only returned if track_history=True. Contains:
        - 'all_paths': list of dicts, each containing the optimization path from a start point
        - 'best_path_index': index of the path that led to the best solution
    """
    if initial_guess is None:
        initial_guess = np.array([1.0, 1.0, 0.0])

    # Run core optimization
    if track_history:
        result, all_paths = _optimize_stress_tensor(
            S_m_hat,
            wavelengths,
            C_values,
            nu,
            L,
            S_i_hat,
            initial_guess,
            max_fringes,
            callback=None,
            track_all_paths=True,
        )

        # Find which path was the best
        best_path_index = None
        for i, path in enumerate(all_paths):
            if path["is_best"]:
                best_path_index = i
                break

        history = {"all_paths": all_paths, "best_path_index": best_path_index}
        return result.x, result.success, history
    else:
        result = _optimize_stress_tensor(
            S_m_hat,
            wavelengths,
            C_values,
            nu,
            L,
            S_i_hat,
            initial_guess,
            max_fringes,
            callback=None,
            track_all_paths=False,
        )
        return result.x, result.success


def recover_stress_tensor_live(
    S_m_hat, wavelengths, C_values, nu, L, S_i_hat, initial_guess=None, update_interval=5, max_fringes=6
):
    """
    Recover stress tensor with live plotting of optimization progress.

    This function is useful for debugging and understanding the optimization process
    for a single pixel. It creates a live-updating plot that shows how the stress
    components and predicted Stokes parameters evolve during optimization.

    Parameters
    ----------
    S_m_hat : ndarray
        Measured normalised Stokes components, shape (3, 2) for RGB channels.
    wavelengths : array-like
        Wavelengths for R, G, B channels (m).
    C_values : array-like
        Stress-optic coefficients for R, G, B channels (1/Pa).
    nu : float
        Solid fraction (use 1.0 for solid samples).
    L : float
        Sample thickness (m).
    S_i_hat : array-like
        Incoming normalised Stokes vector [S1_hat, S2_hat] or [S1_hat, S2_hat, S3_hat].
    initial_guess : array-like, optional
        Initial guess for stress tensor [sigma_xx, sigma_yy, sigma_xy].
    update_interval : int, optional
        Update plot every N iterations. Default is 5.
    max_fringes : float, optional
        Maximum expected fringe order for setting bounds. Default is 6.

    Returns
    -------
    stress_tensor : ndarray
        Recovered stress tensor components [sigma_xx, sigma_yy, sigma_xy].
    success : bool
        Whether optimization was successful.
    history : dict
        Optimization history for further analysis.
    fig : matplotlib.figure.Figure
        The figure object (will be kept open).
    """
    import matplotlib.pyplot as plt

    from photoelastimetry.plotting import plot_optimization_history_live

    if initial_guess is None:
        initial_guess = np.array([1.0, 1.0, 0.0])

    # Storage for tracking
    history = {"stress_params": [], "S_predicted": [], "residuals": []}
    fig, axes = None, None
    iteration_count = [0]  # Use list to allow modification in nested function

    def callback_func(xk):
        """Callback with live plotting."""
        # Store current state
        history["stress_params"].append(xk.copy())

        S_pred_all = np.zeros((3, 2))
        for c in range(3):
            S_pred_all[c] = predict_stokes(xk[0], xk[1], xk[2], C_values[c], nu, L, wavelengths[c], S_i_hat)
        history["S_predicted"].append(S_pred_all.copy())

        residual = compute_residual(xk, S_m_hat, wavelengths, C_values, nu, L, S_i_hat)
        history["residuals"].append(residual)

        # Update plot periodically
        iteration_count[0] += 1
        if iteration_count[0] % update_interval == 0 or iteration_count[0] == 1:
            nonlocal fig, axes
            # Convert to arrays for plotting
            hist_for_plot = {
                "stress_params": np.array(history["stress_params"]),
                "S_predicted": np.array(history["S_predicted"]),
                "residuals": np.array(history["residuals"]),
            }
            fig, axes = plot_optimization_history_live(hist_for_plot, S_m_hat, fig, axes)
            plt.pause(0.01)

    # Run core optimization with live plotting callback
    result = _optimize_stress_tensor(
        S_m_hat, wavelengths, C_values, nu, L, S_i_hat, initial_guess, max_fringes, callback=callback_func
    )

    # Final update
    history["stress_params"] = np.array(history["stress_params"])
    history["S_predicted"] = np.array(history["S_predicted"])
    history["residuals"] = np.array(history["residuals"])

    fig, axes = plot_optimization_history_live(history, S_m_hat, fig, axes)
    plt.ioff()  # Turn off interactive mode

    return result.x, result.success, history, fig


def compute_solid_fraction(S0, S_ref, mu, L):
    """
    Compute solid fraction from intensity using Beer-Lambert law.

    Parameters
    ----------
    S0 : array-like
        Measured intensity (from colour channel with absorptive dye).
    S_ref : float
        Reference light intensity before passing through sample.
    mu : float
        Absorption coefficient for the colour channel (calibrated parameter).
    L : float
        Sample thickness (m).

    Returns
    -------
    nu : array-like
        Solid fraction values.
    """
    # Beer-Lambert: S0 = S_ref * exp(-mu * nu * L)
    # Solving for nu: nu = -ln(S0 / S_ref) / (mu * L)
    S0_safe = np.maximum(S0, 1e-10)
    nu = -np.log(S0_safe / S_ref) / (mu * L)
    return nu


def _process_pixel(args):
    """
    Process a single pixel to recover stress tensor.

    Helper function for parallel processing in recover_stress_map.
    """
    if len(args) == 9:
        y, x, image_stack, wavelengths, C_values, nu, L, S_i_hat, initial_guess = args
    else:
        y, x, image_stack, wavelengths, C_values, nu, L, S_i_hat = args
        initial_guess = None

    # Get intensity measurements for all colour channels
    S_m_hat = np.zeros((3, 2))

    for c in range(3):  # R, G, B
        I = image_stack[y, x, c, :]
        # Skip if any NaN values
        if np.isnan(I).any():
            return (y, x, np.array([np.nan, np.nan, np.nan]))

        # Compute Stokes components
        S0, S1, S2 = compute_stokes_components(I[0], I[1], I[2], I[3])

        # Compute normalised Stokes components
        S1_hat, S2_hat = compute_normalised_stokes(S0, S1, S2)

        S_m_hat[c, 0] = S1_hat
        S_m_hat[c, 1] = S2_hat

    # Get porosity value for this pixel
    nu_pixel = nu if np.isscalar(nu) else nu[y, x]

    # Recover stress tensor
    stress_tensor, success = recover_stress_tensor(
        S_m_hat, wavelengths, C_values, nu_pixel, L, S_i_hat, initial_guess=initial_guess
    )

    if success:
        return (y, x, stress_tensor)
    else:
        return (y, x, np.array([np.nan, np.nan, np.nan]))


def recover_stress_map_stokes(
    image_stack,
    wavelengths,
    C_values,
    nu,
    L,
    S_i_hat,
    initial_guess_map=None,
    n_jobs=-1,
):
    """
    Recover full 2D stress tensor map from polarimetric image stack using Stokes method.

    Parameters
    ----------
    image_stack : ndarray
        Image stack of shape [H, W, 3, 4] where:
        - H, W are image dimensions
        - 3 colour channels (R, G, B)
        - 4 polarisation angles (0, 45, 90, 135 degrees)
    wavelengths : array-like
        Wavelengths for R, G, B channels (m).
    C_values : array-like
        Stress-optic coefficients for R, G, B channels (1/Pa).
    nu : float or ndarray
        Solid fraction. Use 1.0 for solid samples.
        Can be scalar or array matching image dimensions.
    L : float
        Sample thickness (m).
    S_i_hat : array-like
        Incoming normalised Stokes vector [S1_hat, S2_hat] or [S1_hat, S2_hat, S3_hat].
    initial_guess_map : ndarray, optional
        Initial guess stress map [H, W, 3].
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all available cores (default: -1).

    Returns
    -------
    stress_map : ndarray
        Array of shape [H, W, 3] containing [sigma_xx, sigma_yy, sigma_xy] in Pa.
    """
    from joblib import Parallel, delayed

    H, W, _, _ = image_stack.shape
    stress_map = np.zeros((H, W, 3), dtype=np.float32)

    # Create list of all pixel coordinates
    pixel_coords = [(y, x) for y in range(H) for x in range(W)]

    # Create arguments for each pixel
    if initial_guess_map is not None:
        pixel_args = [
            (y, x, image_stack, wavelengths, C_values, nu, L, S_i_hat, initial_guess_map[y, x])
            for y, x in pixel_coords
        ]
    else:
        pixel_args = [(y, x, image_stack, wavelengths, C_values, nu, L, S_i_hat) for y, x in pixel_coords]

    # Process pixels in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_pixel)(args) for args in tqdm(pixel_args, desc="Processing pixels")
    )

    # Fill in the stress map
    for y, x, stress_tensor in results:
        stress_map[y, x, :] = stress_tensor

    return stress_map
