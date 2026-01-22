"""
Phase decomposed seeding for stress initialization.

This module implements the phase decomposed seeding method described in the paper
to provide accurate initial estimates for stress optimization algorithms.
"""

import numpy as np

from photoelastimetry.image import compute_normalised_stokes, compute_stokes_components


def invert_wrapped_retardance(S_m_hat):
    """
    Invert measured Stokes parameters to get wrapped retardance and orientation.

    Parameters
    ----------
    S_m_hat : ndarray
        Measured normalised Stokes parameters [S1, S2] for each channel.
        Shape: (..., n_wavelengths, 2)

    Returns
    -------
    theta : float
        Principal stress orientation in radians [-pi/4, pi/4].
    delta_wrap : ndarray
        Wrapped retardance [0, pi] for each channel. Shape: (..., n_wavelengths)
    """

    s1 = S_m_hat[..., 0]
    s2 = S_m_hat[..., 1]

    # Use vector averaging for angle to handle wrap-around and weight by signal strength
    # 2*theta = atan2(1-s1, s2)
    # Vectors are v = [s2, 1-s1] (x, y)
    # Summing vectors weighs them by their magnitude (approx sin^2(delta/2))
    # Sum over wavelengths (last axis of s1/s2)
    x_sum = np.sum(s2, axis=-1)
    y_sum = np.sum(1 - s1, axis=-1)

    theta_mean = 0.5 * np.arctan2(y_sum, x_sum)

    # Wrapped retardance
    # The magnitude is modulated by sin(2*theta) in a linear polariscope
    # sin^2(delta/2) * |sin(2*theta)| = sqrt((1-S1)^2 + S2^2) / 2
    sin_2theta = np.abs(np.sin(2 * theta_mean))

    raw_magnitude = np.sqrt((1 - s1) ** 2 + s2**2) / 2

    # Avoid division by zero at isoclinics
    # sin_2theta needs to broadcast against raw_magnitude (..., n_wl)
    # theta_mean is (...,) so sin_2theta is (...,)
    sin_2theta_expanded = sin_2theta[..., np.newaxis]

    # Use where to handle safe division
    sin_sq_delta_2 = np.divide(raw_magnitude, sin_2theta_expanded, where=(sin_2theta_expanded > 1e-3))
    # If too close to isoclinic, fallback to raw_magnitude (or just 0 really, but raw is safer)
    sin_sq_delta_2 = np.where(sin_2theta_expanded <= 1e-3, raw_magnitude, sin_sq_delta_2)

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
        Wrapped retardance for each wavelength. Shape (..., n_wavelengths)
    theta : float or ndarray
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
    delta_sigma : float or ndarray
        Estimated stress difference.
    """
    # Ensure inputs are arrays
    delta_wrap = np.asarray(delta_wrap)
    wavelengths = np.asarray(wavelengths)
    C_values = np.asarray(C_values)

    n_channels = wavelengths.shape[0]
    base_shape = delta_wrap.shape[:-1]

    # Flatten pixel dimensions for vectorized processing
    if len(base_shape) > 0:
        n_pixels = np.prod(base_shape)
        d_wrap_flat = delta_wrap.reshape(n_pixels, n_channels)
    else:
        n_pixels = 1
        d_wrap_flat = delta_wrap.reshape(1, n_channels)

    # Calculate factor for delta -> sigma conversion
    factor = wavelengths / (2 * np.pi * C_values * nu * L)

    # Pre-generate search strategies for one channel
    # Strategy = (shift_value, sign_multiplier) where stress ~ factor * (shift + sign*delta)
    # Pos branch: 2*pi*n + delta  -> shift=2*pi*n, sign=1
    # Neg branch: 2*pi*(n+1) - delta -> shift=2*pi*(n+1), sign=-1

    n_vals = np.arange(n_max + 1)
    # Lists of (shift, sign) tuples
    single_channel_strategies = []

    for n in n_vals:
        single_channel_strategies.append((2 * np.pi * n, 1.0))  # Positive branch
    for n in n_vals:
        single_channel_strategies.append((2 * np.pi * (n + 1), -1.0))  # Negative branch

    # We need a strategy choice for each channel
    # strategies per channel
    channel_strategies = [single_channel_strategies] * n_channels

    import itertools

    best_var = np.full(n_pixels, np.inf)
    best_delta_sigma = np.zeros(n_pixels)

    # Iterate over all combinations of strategies across channels
    # total combinations = (2*(n_max+1))^3. For n_max=6, ~2700 combos.
    for combo in itertools.product(*channel_strategies):
        # combo is tuple of (shift, sign) for each channel

        # Calculate stress for all pixels for this strategy combination
        # stress shape: (n_pixels, n_channels)
        stresses = np.zeros((n_pixels, n_channels))

        for c in range(n_channels):
            shift, sign = combo[c]
            stresses[:, c] = factor[c] * (shift + sign * d_wrap_flat[:, c])

        # Filter by sigma_max
        if sigma_max is not None:
            # Check if all channels are within reasonable bounds
            valid_mask = np.all(stresses < sigma_max * 1.5, axis=1)
        else:
            valid_mask = np.ones(n_pixels, dtype=bool)

        if not np.any(valid_mask):
            continue

        # Compute variance for estimating consistency
        # Calculate only for valid pixels to save time?
        # Numpy is fast enough to do all, then mask.
        variance = np.var(stresses, axis=1)
        median_stress = np.median(stresses, axis=1)

        # Update best
        update_locs = valid_mask & (variance < best_var)

        if np.any(update_locs):
            best_var[update_locs] = variance[update_locs]
            best_delta_sigma[update_locs] = median_stress[update_locs]

    # Reshape back to original shape
    if len(base_shape) > 0:
        return best_delta_sigma.reshape(base_shape)
    else:
        return best_delta_sigma.item()


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
        Incoming normalised Stokes vector.
    sigma_max : float, optional
        Maximum allowed stress difference (Pa).
    n_max : int
        Maximum fringe order to search.
    n_jobs : int
        Deprecated. Number of parallel jobs (no longer used due to vectorization).

    Returns
    -------
    stress_map : ndarray
        Initial stress map [H, W, 3].
    """
    # Flatten data for processing
    H, W = data.shape[:2]

    # Compute normalised Stokes for all pixels
    # Function signature: compute_normalised_stokes(data, S_i_hat) -> S_m_hat (H, W, n_wl, 2)
    # Note: image.compute_normalised_stokes expects specific data format.
    # Assuming standard format (H, W, n_wl, n_angles)

    I_0 = data[..., 0]
    I_45 = data[..., 1]
    I_90 = data[..., 2]
    I_135 = data[..., 3]

    S0, S1, S2 = compute_stokes_components(I_0, I_45, I_90, I_135)
    S1_hat, S2_hat = compute_normalised_stokes(S0, S1, S2)
    S_m_hat = np.stack([S1_hat, S2_hat], axis=-1)

    # Reshape for vectorized processing
    S_flat = S_m_hat.reshape(-1, S_m_hat.shape[-2], S_m_hat.shape[-1])

    # Vectorized inversion
    theta, delta_wrap = invert_wrapped_retardance(S_flat)

    # Vectorized fringe resolution
    delta_sigma = resolve_fringe_orders(delta_wrap, theta, wavelengths, C_values, nu, L, sigma_max, n_max)

    # Vectorized stress tensor construction
    # initialize_stress_tensor returns (3, N)
    stress_components = initialize_stress_tensor(delta_sigma, theta)

    # Reshape to (H, W, 3)
    stress_map = np.moveaxis(stress_components, 0, -1).reshape(H, W, 3)

    return stress_map
