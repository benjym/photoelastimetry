"""
Phase decomposed seeding for stress initialization.

This module implements the phase decomposed seeding method described in the paper
to provide accurate initial estimates for stress optimization algorithms.
"""

import numpy as np
from joblib import Parallel, delayed

from photoelastimetry.image import compute_normalized_stokes, compute_stokes_components


def invert_wrapped_retardance(S_m_hat):
    """
    Invert measured Stokes parameters to get wrapped retardance and orientation.

    Parameters
    ----------
    S_m_hat : ndarray
        Measured normalized Stokes parameters [S1, S2] for each channel.
        Shape: (n_wavelengths, 2)

    Returns
    -------
    theta : float
        Principal stress orientation in radians [-pi/4, pi/4].
    delta_wrap : ndarray
        Wrapped retardance [0, pi] for each channel. Shape: (n_wavelengths,)
    """

    s1 = S_m_hat[:, 0]
    s2 = S_m_hat[:, 1]

    # Use vector averaging for angle to handle wrap-around and weight by signal strength
    # 2*theta = atan2(1-s1, s2)
    # Vectors are v = [s2, 1-s1] (x, y)
    # Summing vectors weighs them by their magnitude (approx sin^2(delta/2))
    x_sum = np.sum(s2)
    y_sum = np.sum(1 - s1)

    theta_mean = 0.5 * np.arctan2(y_sum, x_sum)

    # Wrapped retardance
    # The magnitude is modulated by sin(2*theta) in a linear polariscope
    # sin^2(delta/2) * |sin(2*theta)| = sqrt((1-S1)^2 + S2^2) / 2
    sin_2theta = np.abs(np.sin(2 * theta_mean))

    raw_magnitude = np.sqrt((1 - s1) ** 2 + s2**2) / 2

    # Avoid division by zero at isoclinics
    if sin_2theta > 1e-3:
        sin_sq_delta_2 = raw_magnitude / sin_2theta
    else:
        # At isoclinics, we cannot recover retardance reliably.
        # Fallback or keep as 0 (effectively what raw_magnitude gives, scaled)
        sin_sq_delta_2 = raw_magnitude

    # Clamp to [0, 1] for numerical stability
    sin_sq_delta_2 = np.clip(sin_sq_delta_2, 0, 1)

    # delta = 2 * asin(sqrt(...))
    delta_wrap = 2 * np.arcsin(np.sqrt(sin_sq_delta_2))

    return theta_mean, delta_wrap


def resolve_fringe_orders(delta_wrap, theta, wavelengths, C_values, nu, L, sigma_max=None, n_max=6):
    """
    Resolve fringe orders using multi-wavelength consistency.

    Parameters
    ----------
    delta_wrap : ndarray
        Wrapped retardance for each wavelength.
    theta : float
        Orientation angle.
    wavelengths : ndarray
        Wavelengths in meters.
    C_values : ndarray
        Stress-optic coefficients.
    nu : float
        Solid fraction.
    L : float
        Thickness.
    sigma_max : float, optional
        Maximum expected stress difference (Pa).
    n_max : int
        Maximum fringe order to search.

    Returns
    -------
    delta_sigma : float
        Estimated stress difference.
    """
    n_channels = len(wavelengths)

    # Stress proxy factor: x = 2*pi*C*nu*L * delta_sigma
    # delta = (2*pi*C*nu*L / lambda) * delta_sigma
    # delta_sigma = lambda * delta / (2*pi*C*nu*L)
    # delta_total = delta_wrap + 2*pi*n

    # Convert inputs to numpy arrays to ensure element-wise operations work
    wavelengths = np.asarray(wavelengths)
    C_values = np.asarray(C_values)

    # Calculate factor for delta -> sigma conversion
    factor = wavelengths / (2 * np.pi * C_values * nu * L)

    best_var = float("inf")
    best_delta_sigma = 0.0

    # Generate candidates using both phase branches
    # D_pos = 2*pi*n + delta_wrap
    # D_neg = 2*pi*(n+1) - delta_wrap (assuming delta is positive and delta_wrap in [0, pi])
    channel_candidates = []
    import itertools

    for c in range(n_channels):
        n_vals = np.arange(n_max + 1)
        d_pos = 2 * np.pi * n_vals + delta_wrap[c]
        d_neg = 2 * np.pi * (n_vals + 1) - delta_wrap[c]

        cands = np.concatenate([d_pos, d_neg])
        stress_cands = factor[c] * cands

        # Filter individually by max stress
        if sigma_max is not None:
            stress_cands = stress_cands[stress_cands < sigma_max * 1.5]  # Allow some buffer
        channel_candidates.append(stress_cands)

    if not all(len(c) > 0 for c in channel_candidates):
        return 0.0

    best_var = float("inf")
    best_delta_sigma = 0.0

    # Search combinations
    for stress_set in itertools.product(*channel_candidates):
        stress_vals = np.array(stress_set)
        var = np.var(stress_vals)
        if var < best_var:
            best_var = var
            best_delta_sigma = np.median(stress_vals)

    return best_delta_sigma


def initialize_stress_tensor(delta_sigma, theta):
    """
    Initialize stress tensor components assuming granular material (sigma_2 approx 0).

    Parameters
    ----------
    delta_sigma : float
        Stress difference.
    theta : float
        Orientation.

    Returns
    -------
    stress : ndarray
        Stress components [sigma_xx, sigma_yy, sigma_xy].
    """
    # Eqs. 11-13
    # sigma_xx = (delta_sigma/2) * (1 + cos(2*theta))
    # sigma_yy = (delta_sigma/2) * (1 - cos(2*theta))
    # sigma_xy = (delta_sigma/2) * sin(2*theta)

    s_mean = delta_sigma / 2

    sigma_xx = s_mean * (1 + np.cos(2 * theta))
    sigma_yy = s_mean * (1 - np.cos(2 * theta))
    sigma_xy = s_mean * np.sin(2 * theta)

    return np.array([sigma_xx, sigma_yy, sigma_xy])


def _process_pixel(s_hat, wavelengths, C_values, nu, L, sigma_max, n_max):
    """Helper for parallel processing."""
    theta, delta_wrap = invert_wrapped_retardance(s_hat)
    delta_sigma = resolve_fringe_orders(delta_wrap, theta, wavelengths, C_values, nu, L, sigma_max, n_max)
    return initialize_stress_tensor(delta_sigma, theta)


def phase_decomposed_seeding(
    data,
    wavelengths,
    C_values,
    nu,
    L,
    S_i_hat,
    sigma_max=None,
    n_max=6,
    n_jobs=-1,
):
    """
    Compute initial stress guess using phase decomposed seeding method.

    Parameters
    ----------
    data : ndarray
        Input image data.
    wavelengths : ndarray
        Wavelengths in meters.
    C_values : ndarray
        Stress-optic coefficients.
    nu : float
        Solid fraction.
    L : float
        Sample thickness in meters.
    S_i_hat : ndarray
        Incoming normalized Stokes vector.
    sigma_max : float, optional
        Maximum allowed stress difference (Pa).
    n_max : int
        Maximum fringe order to search.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    stress_map : ndarray
        Initial stress map [H, W, 3].
    """
    # Flatten data for processing
    H, W = data.shape[:2]

    # Compute normalized Stokes for all pixels
    # Function signature: compute_normalized_stokes(data, S_i_hat) -> S_m_hat (H, W, n_wl, 2)
    # Note: image.compute_normalized_stokes expects specific data format.
    # Assuming standard format (H, W, n_wl, n_angles)

    I_0 = data[..., 0]
    I_45 = data[..., 1]
    I_90 = data[..., 2]
    I_135 = data[..., 3]

    S0, S1, S2 = compute_stokes_components(I_0, I_45, I_90, I_135)
    S1_hat, S2_hat = compute_normalized_stokes(S0, S1, S2)
    S_m_hat = np.stack([S1_hat, S2_hat], axis=-1)

    # Reshape for parallel processing
    S_flat = S_m_hat.reshape(-1, S_m_hat.shape[-2], S_m_hat.shape[-1])

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_pixel)(s, wavelengths, C_values, nu, L, sigma_max, n_max) for s in S_flat
    )

    stress_map = np.array(results).reshape(H, W, 3)
    return stress_map
