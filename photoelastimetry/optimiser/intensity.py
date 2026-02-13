"""
Intensity-based stress measurement using raw polarimetric intensities.

This module implements stress tensor recovery by optimizing directly on raw
polarization intensity measurements rather than normalised Stokes components.
This approach:

1. Avoids information loss from normalisation
2. Eliminates angle-wrapping ambiguities from Stokes → angle conversion
3. Allows proper statistical modeling of detector noise (Poisson + Gaussian)
4. Enables joint calibration of instrument parameters
5. Provides more robust inversion near degenerate points (low retardance)

The forward model predicts intensity for each analyzer angle and wavelength
from the stress tensor via Mueller/Jones calculus.
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from tqdm import tqdm

from photoelastimetry.image import (
    compute_normalised_stokes,
    compute_principal_angle,
    compute_retardance,
    compute_stokes_components,
    mueller_matrix,
)


def predict_intensity(
    sigma_xx,
    sigma_yy,
    sigma_xy,
    C,
    nu,
    L,
    wavelength,
    analyzer_angles,
    S_i_hat,
    I0=1.0,
):
    """
    Predict intensity measurements for different analyzer angles.

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
    analyzer_angles : array-like
        Analyzer angles in radians [0, π/4, π/2, 3π/4].
    S_i_hat : array-like
        Incoming normalised Stokes vector [S1_hat, S2_hat, S3_hat].
    I0 : float, optional
        Incident intensity (default: 1.0).

    Returns
    -------
    intensities : ndarray
        Predicted intensities for each analyzer angle.

    Notes
    -----
    The intensity at analyzer angle α is computed using Mueller calculus:
        I(α) = [1, cos(2α), sin(2α), 0] @ M_sample @ S_in

    where M_sample is the Mueller matrix of the birefringent sample.
    """
    # Compute retardance and fast axis orientation from stress
    theta = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)
    delta = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength)

    # Get sample Mueller matrix
    M = mueller_matrix(theta, delta)

    # Incident Stokes vector from S_i_hat
    # S_in = I0 * [1, cos(2*pol_angle), sin(2*pol_angle), 0]
    S_i_hat = np.asarray(S_i_hat)
    if S_i_hat.ndim == 0:
        # Scalar?? Should not happen but fallback
        S_in = I0 * np.array([1.0, 1.0, 0.0, 0.0])  # Default Linear Horizontal
    elif len(S_i_hat) == 2:
        S_in = I0 * np.array([1.0, S_i_hat[0], S_i_hat[1], 0.0])
    else:
        S_in = I0 * np.array([1.0, S_i_hat[0], S_i_hat[1], S_i_hat[2]])

    # Transmitted Stokes vector
    S_out = M @ S_in

    # Measure intensity through analyzer at each angle
    # Analyzer at angle α transmits: I = (S0 + S1*cos(2α) + S2*sin(2α)) / 2
    intensities = np.zeros(len(analyzer_angles))
    for i, alpha in enumerate(analyzer_angles):
        cos_2a = np.cos(2 * alpha)
        sin_2a = np.sin(2 * alpha)
        # Intensity = (S0 + S1*cos(2α) + S2*sin(2α)) / 2
        intensities[i] = 0.5 * (S_out[0] + S_out[1] * cos_2a + S_out[2] * sin_2a)

    return intensities


def compute_intensity_residual(
    stress_params,
    I_measured,
    wavelengths,
    C_values,
    nu,
    L,
    analyzer_angles,
    S_i_hat,
    I0=1.0,
    weights=None,
):
    """
    Compute residual between measured and predicted intensities.

    Parameters
    ----------
    stress_params : array-like
        Stress tensor components [sigma_xx, sigma_yy, sigma_xy].
    I_measured : ndarray
        Measured intensities, shape (n_wavelengths, n_angles).
        Typically (3, 4) for RGB × 4 analyzer angles.
    wavelengths : array-like
        Wavelengths for each channel (m).
    C_values : array-like
        Stress-optic coefficients for each channel (1/Pa).
    nu : float
        Solid fraction (use 1.0 for solid samples).
    L : float
        Sample thickness (m).
    analyzer_angles : array-like
        Analyzer angles in radians [0, π/4, π/2, 3π/4].
    S_i_hat : array-like
        Incoming normalised Stokes vector [S1_hat, S2_hat, S3_hat].
    I0 : float, optional
        Incident intensity (default: 1.0).
    weights : ndarray, optional
        Weights for residuals, shape matching I_measured.
        If None, uniform weighting is used.

    Returns
    -------
    residual : ndarray
        Flattened array of weighted residuals.
    """
    sigma_xx, sigma_yy, sigma_xy = stress_params

    residuals = []

    for c, (wavelength, C) in enumerate(zip(wavelengths, C_values)):
        I_pred = predict_intensity(
            sigma_xx,
            sigma_yy,
            sigma_xy,
            C,
            nu,
            L,
            wavelength,
            analyzer_angles,
            S_i_hat,
            I0,
        )

        diff = I_measured[c] - I_pred

        # Apply weights if provided (for Poisson noise: weight ~ 1/sqrt(I))
        if weights is not None:
            diff *= weights[c]

        residuals.append(diff)

    return np.concatenate(residuals)


def recover_stress_tensor_intensity(
    I_measured,
    wavelengths,
    C_values,
    nu,
    L,
    S_i_hat,
    analyzer_angles=None,
    I0=1.0,
    weights=None,
    initial_guess=None,
    method="lm",
    bounds=None,
):
    """
    Recover stress tensor from raw intensity measurements.

    This function inverts the forward model to find the stress tensor that
    best explains the measured intensities across all wavelengths and analyzer
    angles.

    Parameters
    ----------
    I_measured : ndarray
        Measured intensities, shape (n_wavelengths, n_angles).
        Typically (3, 4) for RGB × 4 analyzer angles [0°, 45°, 90°, 135°].
    wavelengths : array-like
        Wavelengths for R, G, B channels (m).
    C_values : array-like
        Stress-optic coefficients for R, G, B channels (1/Pa).
    nu : float
        Solid fraction (use 1.0 for solid samples).
    L : float
        Sample thickness (m).
    analyzer_angles : array-like, optional
        Analyzer angles in radians. Default: [0, π/4, π/2, 3π/4].
    S_i_hat : array-like
        Incoming normalised Stokes vector [S1_hat, S2_hat, S3_hat].
    I0 : float, optional
        Incident intensity for normalisation (default: 1.0).
    weights : ndarray, optional
        Weights for residuals, shape matching I_measured.
        For Poisson noise, use weights[c, i] = 1/sqrt(I_measured[c, i]).
        If None, uniform weighting is used.
    initial_guess : array-like, optional
        Initial guess for stress tensor [sigma_xx, sigma_yy, sigma_xy].
        If None, computed from Stokes-based quick estimate.
    method : str, optional
        Optimization method: 'lm' (Levenberg-Marquardt, default),
        'trf', 'dogbox', or 'nelder-mead'.
    bounds : tuple of array-like, optional
        Lower and upper bounds for parameters (lower, upper).
        Each should be array of length 3 for [sigma_xx, sigma_yy, sigma_xy].
        Default: (-1e9, 1e9) Pa for all components.

    Returns
    -------
    stress_tensor : ndarray
        Recovered stress tensor components [sigma_xx, sigma_yy, sigma_xy].
    success : bool
        Whether optimization was successful.
    result : OptimizeResult
        Full optimization result object.

    Notes
    -----
    This method is generally more accurate than Stokes-based inversion when:
    - High precision is required
    - Working near zero retardance (degenerate cases)
    - Principal stress axes aligned with incident polarization
    - Want to model detector noise properly (Poisson statistics)

    For typical usage, the Stokes-based method is faster and often sufficient.

    Examples
    --------
    >>> # Typical usage with 3 wavelengths, 4 analyzer angles
    >>> I_measured = np.array([[I0_R, I45_R, I90_R, I135_R],
    ...                        [I0_G, I45_G, I90_G, I135_G],
    ...                        [I0_B, I45_B, I90_B, I135_B]])
    >>> wavelengths = np.array([650e-9, 550e-9, 450e-9])
    >>> C_values = np.array([2e-12, 2e-12, 2e-12])
    >>> stress, success, result = recover_stress_tensor_intensity(
    ...     I_measured, wavelengths, C_values, nu=1.0, L=0.01
    ... )
    """
    if analyzer_angles is None:
        analyzer_angles = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])

    # Compute initial guess from Stokes-based method if not provided
    if initial_guess is None:
        # Quick estimate: compute Stokes and use closed-form estimate
        from photoelastimetry.optimiser.stokes import recover_stress_tensor

        S_m_hat = np.zeros((len(wavelengths), 2))

        for c in range(len(wavelengths)):
            I_data = I_measured[c]
            # compute_stokes_components expects: I_0, I_45, I_90, I_135
            # Our analyzer_angles are [0, π/4, π/2, 3π/4] so ordering matches
            if len(I_data) >= 4:
                S0, S1, S2 = compute_stokes_components(I_data[0], I_data[1], I_data[2], I_data[3])
                S1_hat, S2_hat = compute_normalised_stokes(S0, S1, S2)
                S_m_hat[c, 0] = S1_hat
                S_m_hat[c, 1] = S2_hat

        # Use Stokes method for initial guess
        initial_guess, _ = recover_stress_tensor(S_m_hat, wavelengths, C_values, nu, L, S_i_hat)

    # Set bounds if not provided
    if bounds is None:
        bounds = (np.array([-1e9, -1e9, -1e9]), np.array([1e9, 1e9, 1e9]))

    # Choose optimization method
    if method == "nelder-mead":
        # Use Nelder-Mead for robustness (no gradient needed)
        result = minimize(
            lambda x: np.sum(
                compute_intensity_residual(
                    x,
                    I_measured,
                    wavelengths,
                    C_values,
                    nu,
                    L,
                    analyzer_angles,
                    S_i_hat,
                    I0,
                    weights,
                )
                ** 2
            ),
            initial_guess,
            method="Nelder-Mead",
            options={"xatol": 1e-16, "fatol": 1e-16, "maxiter": 2000},
        )
        stress_tensor = result.x
        success = result.success

    else:
        # Use least_squares for methods that can exploit Jacobian
        # Note: 'lm' method doesn't support bounds
        if method == "lm":
            result = least_squares(
                compute_intensity_residual,
                initial_guess,
                args=(
                    I_measured,
                    wavelengths,
                    C_values,
                    nu,
                    L,
                    analyzer_angles,
                    S_i_hat,
                    I0,
                    weights,
                ),
                method=method,
                ftol=1e-8,
                xtol=1e-8,
                max_nfev=1000,
            )
        else:
            result = least_squares(
                compute_intensity_residual,
                initial_guess,
                args=(
                    I_measured,
                    wavelengths,
                    C_values,
                    nu,
                    L,
                    analyzer_angles,
                    S_i_hat,
                    I0,
                    weights,
                ),
                method=method,
                bounds=bounds,
                ftol=1e-8,
                xtol=1e-8,
                max_nfev=1000,
            )
        stress_tensor = result.x
        success = result.success

    return stress_tensor, success, result


def recover_stress_map_intensity(
    image_stack,
    wavelengths,
    C_values,
    nu,
    L,
    S_i_hat,
    analyzer_angles=None,
    I0=1.0,
    use_poisson_weights=True,
    initial_guess_method="uniform",
    method="nelder-mead",
    initial_guess_map=None,
    n_jobs=-1,
):
    """
    Recover full 2D stress tensor map from raw intensity image stack using intensity method.

    Parameters
    ----------
    image_stack : ndarray
        Image stack of shape [H, W, n_wavelengths, n_angles] where:
        - H, W are image dimensions
        - n_wavelengths is number of color channels (typically 3 for RGB)
        - n_angles is number of polarization analyzer angles (typically 4)
    wavelengths : array-like
        Wavelengths for each channel (m).
    C_values : array-like
        Stress-optic coefficients for each channel (1/Pa).
    nu : float or ndarray
        Solid fraction. Use 1.0 for solid samples.
        Can be scalar or array matching image dimensions [H, W].
    L : float
        Sample thickness (m).
    analyzer_angles : array-like, optional
        Analyzer angles in radians. Default: [0, π/4, π/2, 3π/4].
    S_i_hat : array-like
        Incoming normalised Stokes vector [S1_hat, S2_hat, S3_hat].
    I0 : float, optional
        Reference incident intensity (default: 1.0).
    use_poisson_weights : bool, optional
        Whether to use Poisson noise weighting (weight ~ 1/sqrt(I)).
        Default: True.
    initial_guess_method : str, optional
        Method for initial guess: 'stokes' (default) uses Stokes-based solver,
        'zero' uses zeros, 'uniform' uses small uniform stress.
    method : str, optional
        Optimization method: 'lm' (default), 'trf', 'dogbox', or 'nelder-mead'.
    initial_guess_map : ndarray, optional
        Initial guess stress map [H, W, 3]. Overrides initial_guess_method.
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all available cores (default: -1).

    Returns
    -------
    stress_map : ndarray
        Array of shape [H, W, 3] containing [sigma_xx, sigma_yy, sigma_xy] in Pa.
    success_map : ndarray
        Boolean array of shape [H, W] indicating successful convergence.

    Notes
    -----
    This is the main function for generating stress maps using intensity-based
    inversion. It processes each pixel independently and can run in parallel.

    For most applications, this method is slower but more accurate than the
    Stokes-based method, especially near degenerate points.
    """
    from joblib import Parallel, delayed

    if analyzer_angles is None:
        analyzer_angles = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])

    H, W, n_wavelengths, n_angles = image_stack.shape
    stress_map = np.zeros((H, W, 3), dtype=np.float32)
    success_map = np.zeros((H, W), dtype=bool)

    # Optionally compute initial guess map using Stokes method
    if initial_guess_map is None and initial_guess_method == "stokes":
        print("Computing initial guess using Stokes-based solver...")
        from photoelastimetry.optimiser.stokes import recover_stress_map_stokes

        initial_guess_map = recover_stress_map_stokes(
            image_stack,
            wavelengths,
            C_values,
            nu,
            L,
            S_i_hat,
            n_jobs=n_jobs,
        )

    def process_pixel(y, x):
        """Process a single pixel."""
        # Extract intensities for this pixel
        I_measured = image_stack[y, x, :, :]  # shape: (n_wavelengths, n_angles)

        # Skip if any NaN values
        if np.isnan(I_measured).any():
            return (y, x, np.array([np.nan, np.nan, np.nan]), False)

        # Compute weights for Poisson noise if requested
        weights = None
        if use_poisson_weights:
            # Weight = 1/sqrt(I) for Poisson, but avoid division by zero
            I_safe = np.maximum(I_measured, 1e-6)
            weights = 1.0 / np.sqrt(I_safe)

        # Get solid fraction for this pixel
        nu_pixel = nu if np.isscalar(nu) else nu[y, x]

        # Get initial guess
        if initial_guess_map is not None:
            initial_guess = initial_guess_map[y, x, :]
        elif initial_guess_method == "uniform":
            initial_guess = np.array([1e6, 1e6, 0.0])
        else:  # zero
            initial_guess = np.array([0.0, 0.0, 0.0])

        # Recover stress tensor
        stress_tensor, success, _ = recover_stress_tensor_intensity(
            I_measured,
            wavelengths,
            C_values,
            nu_pixel,
            L,
            S_i_hat,
            analyzer_angles,
            I0,
            weights,
            initial_guess,
            method,
        )

        return (y, x, stress_tensor, success)

    # Create list of pixel coordinates
    pixel_coords = [(y, x) for y in range(H) for x in range(W)]

    # Process pixels in parallel
    print(f"Processing {H}×{W} = {H * W} pixels with intensity-based solver...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_pixel)(y, x) for y, x in tqdm(pixel_coords, desc="Intensity inversion")
    )

    # Fill in the stress map
    for y, x, stress_tensor, success in results:
        stress_map[y, x, :] = stress_tensor
        success_map[y, x] = success

    success_rate = np.sum(success_map) / (H * W) * 100
    print(f"Success rate: {success_rate:.1f}%")

    return stress_map, success_map


def compare_stokes_vs_intensity(
    image_stack,
    wavelengths,
    C_values,
    nu,
    L,
    S_i_hat,
    analyzer_angles=None,
    true_stress=None,
):
    """
    Compare Stokes-based and intensity-based inversion methods.

    This utility function runs both methods on the same data and provides
    comparison metrics. Useful for validation and understanding trade-offs.

    Parameters
    ----------
    image_stack : ndarray
        Image stack [H, W, n_wavelengths, n_angles].
    wavelengths : array-like
        Wavelengths for each channel (m).
    C_values : array-like
        Stress-optic coefficients (1/Pa).
    nu : float or ndarray
        Solid fraction.
    L : float
        Sample thickness (m).
    analyzer_angles : array-like, optional
        Analyzer angles in radians.
    S_i_hat : array-like
        Incoming normalised Stokes vector [S1_hat, S2_hat, S3_hat].
    true_stress : ndarray, optional
        Ground truth stress field [H, W, 3] for validation.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'stokes_stress': Stokes solver result [H, W, 3]
        - 'intensity_stress': Intensity solver result [H, W, 3]
        - 'stokes_success_rate': Convergence rate for Stokes
        - 'intensity_success_rate': Convergence rate for intensity
        - 'stokes_residual': Mean residual for Stokes
        - 'intensity_residual': Mean residual for intensity
        - 'stokes_error': RMSE vs truth (if provided)
        - 'intensity_error': RMSE vs truth (if provided)
        - 'runtime_stokes': Execution time for Stokes
        - 'runtime_intensity': Execution time for intensity
    """
    import time

    from photoelastimetry.optimiser.stokes import recover_stress_map_stokes

    print("=== Running Stokes-based Solver ===")
    t0 = time.time()
    stress_stokes = recover_stress_map_stokes(image_stack, wavelengths, C_values, nu, L, S_i_hat, n_jobs=-1)
    t_stokes = time.time() - t0
    print(f"Completed in {t_stokes:.2f} seconds")

    print("\n=== Running Intensity-based Solver ===")
    t0 = time.time()
    stress_intensity, success_map = recover_stress_map_intensity(
        image_stack,
        wavelengths,
        C_values,
        nu,
        L,
        S_i_hat=S_i_hat,
        analyzer_angles=analyzer_angles,
        use_poisson_weights=True,
        initial_guess_method="stokes",
        n_jobs=-1,
    )
    t_intensity = time.time() - t0
    print(f"Completed in {t_intensity:.2f} seconds")

    # Compute success rates
    stokes_success_rate = (
        np.sum(~np.isnan(stress_stokes[:, :, 0])) / stress_stokes.shape[0] / stress_stokes.shape[1] * 100
    )
    intensity_success_rate = np.sum(success_map) / success_map.size * 100

    results = {
        "stokes_stress": stress_stokes,
        "intensity_stress": stress_intensity,
        "stokes_success_rate": stokes_success_rate,
        "intensity_success_rate": intensity_success_rate,
        "runtime_stokes": t_stokes,
        "runtime_intensity": t_intensity,
    }

    # Compute errors vs ground truth if provided
    if true_stress is not None:
        stokes_error = np.sqrt(np.nanmean((stress_stokes - true_stress) ** 2))
        intensity_error = np.sqrt(np.nanmean((stress_intensity - true_stress) ** 2))
        results["stokes_error"] = stokes_error
        results["intensity_error"] = intensity_error

        print("\n=== Comparison Results ===")
        print(f"Stokes RMSE vs truth: {stokes_error:.3e} Pa")
        print(f"Intensity RMSE vs truth: {intensity_error:.3e} Pa")
        print(f"Improvement: {(1 - intensity_error / stokes_error) * 100:.1f}%")

    print(f"\nStokes success rate: {stokes_success_rate:.1f}%")
    print(f"Intensity success rate: {intensity_success_rate:.1f}%")
    print(f"Stokes runtime: {t_stokes:.2f} s")
    print(f"Intensity runtime: {t_intensity:.2f} s")
    print(f"Speedup: {t_stokes / t_intensity:.2f}x")

    return results
