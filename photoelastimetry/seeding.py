"""
Phase decomposed seeding for stress initialisation.

This module implements the phase decomposed seeding method described in the paper
to provide accurate initial estimates for stress optimization algorithms.
"""

import numpy as np

from photoelastimetry.correction import compute_disorder_correction, estimate_grain_encounters
from photoelastimetry.image import compute_normalised_stokes, compute_stokes_components
from photoelastimetry.unwrapping import unwrap_angles_graph_cut


def invert_wrapped_retardance(S_m_hat, S_i_hat):
    """
    Invert measured Stokes parameters to get wrapped retardance and orientation.

    Parameters
    ----------
    S_m_hat : ndarray
        Measured normalised Stokes parameters [S1, S2] for each channel.
        Shape: (..., n_wavelengths, 2)
    S_i_hat : ndarray
        Input normalised Stokes parameter [S1, S2, S3].

    Returns
    -------
    theta : float
        Principal stress orientation in radians [0, pi/2].
    delta_wrap : ndarray
        Wrapped retardance [0, pi] for each channel. Shape: (..., n_wavelengths)
    """

    s1 = S_m_hat[..., 0]
    s2 = S_m_hat[..., 1]

    # Check for circular polarization input (large S3 component)
    is_circular = abs(S_i_hat[2]) > 0.9

    if is_circular:
        # Circular polarisation implementation
        # Reference: "Advancing instantaneous photoelastic method with color polarization camera", Zhang et al.

        S3_in = S_i_hat[2]

        # Calculate theta
        # General case for S3_in (can be +1 or -1)
        # S1 = S3_in * sin(2theta) * sin(delta)
        # S2 = -S3_in * cos(2theta) * sin(delta)
        #
        # We want to recover 2theta.
        # S1 = -S3_in * sin(2theta) * sin(delta)
        # S2 = S3_in * cos(2theta) * sin(delta)
        #
        # sin(2theta) ~ -S1 / S3_in
        # cos(2theta) ~ S2 / S3_in
        # 2theta = atan2(-S1/S3_in, S2/S3_in) = atan2(-S1*S3_in, S2*S3_in)

        # Sum vectors over wavelengths for robustness
        # Note: arctan2(y, x). y ~ sin(2theta), x ~ cos(2theta)
        # S1 ~ sin(2theta), S2 ~ -cos(2theta)
        # => x (cos) ~ -S2, y (sin) ~ S1

        x_sum = np.sum(-s2 * S3_in, axis=-1)
        y_sum = np.sum(s1 * S3_in, axis=-1)
        theta = 0.5 * np.arctan2(y_sum, x_sum)

        # Calculate delta (wrapped)
        # Magnitude = |sin(delta)| => delta_wrap in [0, pi/2]
        magnitude = np.sqrt(s1**2 + s2**2)
        magnitude = np.clip(magnitude, 0, 1)
        delta_wrap = np.arcsin(magnitude)

        return theta, delta_wrap

    S1_in = S_i_hat[0]
    S2_in = S_i_hat[1]

    # Calculate alpha (input polarisation angle) from Stokes parameters
    alpha = 0.5 * np.arctan2(S2_in, S1_in)

    # Calculate difference vector components
    dx = s2 - S2_in
    dy = S1_in - s1

    # Sum vectors over wavelengths to handle wrap-around and weight by signal strength
    x_sum = np.sum(dx, axis=-1)
    y_sum = np.sum(dy, axis=-1)

    # 2*theta = atan2(Y, X)
    theta_mean = 0.5 * np.arctan2(y_sum, x_sum)
    sin_weight = np.abs(np.sin(2 * (theta_mean - alpha)))
    R = np.sqrt(dx**2 + dy**2)
    raw_magnitude = R / 2.0

    # Avoid division by zero at isoclinics (where W -> 0)
    sin_weight_expanded = sin_weight[..., np.newaxis]

    # Use where to handle safe division
    sin_sq_delta_2 = np.divide(raw_magnitude, sin_weight_expanded, where=(sin_weight_expanded > 1e-3))

    # If too close to isoclinic, fallback or set to 0
    sin_sq_delta_2 = np.where(sin_weight_expanded <= 1e-3, 0.0, sin_sq_delta_2)

    # Clamp to [0, 1] for numerical stability
    sin_sq_delta_2 = np.clip(sin_sq_delta_2, 0, 1)

    delta_wrap = 2 * np.arcsin(np.sqrt(sin_sq_delta_2))

    # theta = 0.5 * np.arctan2(
    #     y_sum * np.mean(np.sin(delta_wrap), axis=-1), x_sum * np.mean(np.sin(delta_wrap), axis=-1)
    # )
    theta = theta_mean  # Using mean theta directly

    return theta, delta_wrap


def resolve_fringe_orders(
    delta_wrap, wavelengths, C_values, nu, L, sigma_max=None, n_max=6, is_circular=False
):
    """
    Resolve fringe orders using multi-wavelength consistency.

    Parameters
    ----------
    delta_wrap : ndarray
        Wrapped retardance for each wavelength. Shape (..., n_wavelengths)
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
    is_circular : bool, optional
        Whether the input data is from a circular polariscope (affects wrapping strategies).

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
    channel_strategies_indices = []
    candidate_stresses = []

    # Pre-calculate all possible stress candidates for each channel
    # This avoids recomputing them N^2 times in the inner loop
    for c in range(n_channels):
        strategies = []
        if is_circular:
            # Circular: k*pi +/- delta
            for k in range(2 * n_max + 2):
                strategies.append((np.pi * k, 1.0))
                if k > 0:
                    strategies.append((np.pi * k, -1.0))
        else:
            # Linear: 2*pi*n +/- delta
            n_vals = np.arange(n_max + 1)
            for n in n_vals:
                strategies.append((2 * np.pi * n, 1.0))
                for n in n_vals:
                    strategies.append((2 * np.pi * (n + 1), -1.0))

        # Remove duplicates if any (though construction avoids them mostly)
        # Calculate candidates for this channel
        # Shape: (n_strategies, n_pixels)
        c_candidates = []
        effective_strategies = []

        for idx, (shift, sign) in enumerate(strategies):
            s = factor[c] * (shift + sign * d_wrap_flat[:, c])
            c_candidates.append(s)
            effective_strategies.append(idx)

        candidate_stresses.append(np.array(c_candidates))
        channel_strategies_indices.append(range(len(effective_strategies)))

    import itertools

    best_metric = np.full(n_pixels, np.inf)
    best_indices = np.zeros((n_pixels, n_channels), dtype=int)

    # Convert to list of arrays for faster indexing
    candidates_arrays = candidate_stresses  # List of (N_strat, N_pixels)

    # Iterate over all combinations of strategy INDICES
    for combo_indices in itertools.product(*channel_strategies_indices):
        # combo_indices is tuple (i, j, k)

        # Retrieve precomputed stresses (no compute here, just lookup)
        # s0, s1, s2...
        current_stresses_list = [candidates_arrays[c][idx] for c, idx in enumerate(combo_indices)]

        # Stack to (n_channels, n_pixels) -> Transpose to (n_pixels, n_channels)
        # Actually keeping as list of arrays is fine for math

        # Check sigma_max constraint
        if sigma_max is not None:
            # Efficient check: all must be < limit
            # AND all must be > 0 (stress difference is positive)
            limit = sigma_max * 1.5
            is_valid = True
            for s in current_stresses_list:
                # We can check validity per pixel
                # But to save doing math on invalid pixels, we compute a mask
                pass

            # Perform validity check vectorized
            # Stack for vectorized operations: shape (n_channels, n_pixels)
            # This stack is fast enough?
            stresses_stack = np.stack(current_stresses_list, axis=0)  # (Channels, Pixels)

            valid_mask = np.all((stresses_stack < limit) & (stresses_stack > -1e-10), axis=0)

            if not np.any(valid_mask):
                continue

            # Only compute metric for valid pixels
            # Metric: Sum of Squared Differences (proportional to variance)
            # SSD = sum((x - mean)^2)
            # For 3 values: (a-b)^2 + (b-c)^2 + (c-a)^2 is proportional to variance

            # Using vectorized variance is fine if masked
            # var(axis=0) on stack

            stresses_valid = stresses_stack[:, valid_mask]
            metric = np.var(stresses_valid, axis=0)

            # Compare with best
            current_best_valid = best_metric[valid_mask]
            improvement_mask = metric < current_best_valid

            # Where we have valid pixels that improve on the metric:
            # We need to map back to original indices
            # valid_mask is size n_pixels (boolean)
            # improvement_mask is size count_nonzero(valid_mask)

            # Indices in the compressed array
            # We want indices in the full array where (valid & improvement)

            # Full boolean mask
            update_mask = np.zeros(n_pixels, dtype=bool)
            update_mask[valid_mask] = improvement_mask

            if np.any(update_mask):
                best_metric[update_mask] = metric[improvement_mask]
                best_indices[update_mask] = combo_indices

        else:
            # Similar logic without bounds check
            stresses_stack = np.stack(current_stresses_list, axis=0)
            metric = np.var(stresses_stack, axis=0)
            update_mask = metric < best_metric
            best_metric[update_mask] = metric[update_mask]
            best_indices[update_mask] = combo_indices

    # Reconstruct best comparison
    # best_indices shape (n_pixels, n_channels)
    final_stresses = np.zeros((n_pixels, n_channels))
    for c in range(n_channels):
        # Advanced indexing: valid strategy index per pixel
        strat_indices = best_indices[:, c]
        # candidate_stresses[c] has shape (N_strat, n_pixels)
        # We need to pick element [strat_indices[p], p] for each pixel p
        # Row indices: strat_indices
        # Col indices: arange(n_pixels)
        final_stresses[:, c] = candidates_arrays[c][strat_indices, np.arange(n_pixels)]

    # Final median
    best_delta_sigma = np.median(final_stresses, axis=1)

    # Reshape back to original shape
    if len(base_shape) > 0:
        return best_delta_sigma.reshape(base_shape)
    else:
        return best_delta_sigma.item()


def initialise_stress_tensor(delta_sigma, theta, K=1):
    """
    Initialise stress tensor components assuming a principal stress ratio K. Under these conditions, the stresses are:
     - sigma_1 = delta_sigma / (1 - K)
     - sigma_2 = K * sigma_1
     - p = (sigma_1 + sigma_2) / 2
     - sigma_xx = p + (delta_sigma / 2) * cos(2*theta)
     - sigma_yy = p - (delta_sigma / 2) * cos(2*theta)
     - sigma_xy = (delta_sigma / 2) * sin(2*theta)

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

    sigma_1 = delta_sigma / (1 - K)
    sigma_2 = K * sigma_1
    p = (sigma_1 + sigma_2) / 2
    cos_2theta = np.cos(2 * theta)
    sin_2theta = np.sin(2 * theta)
    sigma_xx = p + (delta_sigma / 2) * cos_2theta
    sigma_yy = p - (delta_sigma / 2) * cos_2theta
    sigma_xy = (delta_sigma / 2) * sin_2theta

    return np.array([sigma_xx, sigma_yy, sigma_xy])


def phase_decomposed_seeding(
    data,
    wavelengths,
    C_values,
    nu,
    L,
    S_i_hat=None,
    sigma_max=None,
    n_max=6,
    K=0.5,
    correction_params=None,
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
    S_i_hat : ndarray, optional
        Incoming normalised Stokes vector.
    sigma_max : float, optional
        Maximum allowed stress difference (Pa).
    n_max : int
        Maximum fringe order to search.
    K : float, optional. Default=0.5
        Principal stress ratio for initialisation.
    correction_params : dict, optional
        Parameters for disorder correction:
        - enabled (bool): whether to apply correction
        - order_param (float): order parameter |m| [0, 1]
        - N (float, optional): number of encounters
        - d (float, optional): particle diameter (to calculate N)

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

    if S_i_hat is None:
        # Default: linear horizontal polarization
        S_i_hat = np.array([1.0, 0.0, 0.0])

    # Vectorized inversion
    theta, delta_wrap = invert_wrapped_retardance(S_flat, S_i_hat)

    is_circular = abs(S_i_hat[2]) > 0.9

    # Vectorized fringe resolution
    delta_sigma = resolve_fringe_orders(
        delta_wrap, wavelengths, C_values, nu, L, sigma_max, n_max, is_circular=is_circular
    )

    # Unflatten theta and delta_sigma
    theta = theta.reshape(H, W)
    delta_sigma = delta_sigma.reshape(H, W)

    if is_circular:
        # Re-calculate theta robustly using the resolved stress difference
        # This handles the sign ambiguity of sin(delta) for each wavelength

        # Compute theoretical delta for all pixels and wavelengths
        # delta_sigma: (H, W) -> (H, W, 1) to broadcast vs (3,)
        delta_expected = (2 * np.pi * C_values * nu * L * delta_sigma[..., np.newaxis]) / wavelengths

        # Determine sign of sin(delta)
        sign_sin_delta = np.sign(np.sin(delta_expected))

        # Retrieve Stokes components (H, W, n_wl)
        s1 = S_m_hat[..., 0]
        s2 = S_m_hat[..., 1]
        S3_in = S_i_hat[2]

        # Weighted vector sum for 2*theta
        # Based on Mueller matrix in image.py:
        # S1 = S3 * sin(2theta) * sin(delta)
        # S2 = -S3 * cos(2theta) * sin(delta)
        #
        # sin(2theta) = S1 / (S3 * sin(delta)) ~ S1 * S3 * sgn(sin(delta))
        # cos(2theta) = -S2 / (S3 * sin(delta)) ~ -S2 * S3 * sgn(sin(delta))

        X = -s2 * S3_in * sign_sin_delta
        Y = s1 * S3_in * sign_sin_delta

        X_sum = np.sum(X, axis=-1)
        Y_sum = np.sum(Y, axis=-1)

        # 2*theta = atan2(Y, X)
        theta = 0.5 * np.arctan2(Y_sum, X_sum)

        # For stress tensor initialization (uses 2*theta), unwrapping is not strictly needed.
        unwrapped_theta = theta
    # else:
    if correction_params and correction_params.get("unwrap_angles", False):
        # Unwrap angles
        unwrapped_theta = unwrap_angles_graph_cut(theta, quality=delta_sigma)

    # Apply disorder correction if enabled
    if correction_params and correction_params.get("enabled", False):
        order_param = correction_params.get("order_param")

        N = correction_params.get("N")

        if N is None and "d" in correction_params:
            d = correction_params["d"]
            N = estimate_grain_encounters(nu, L, d)

        if N is not None:
            correction_factor = compute_disorder_correction(N, order_param)
            delta_sigma *= correction_factor

    # Vectorized stress tensor construction
    # initialise_stress_tensor returns (3, N)
    stress_components = initialise_stress_tensor(delta_sigma.flatten(), unwrapped_theta.flatten(), K)
    # Reshape to (H, W, 3)
    stress_map = np.moveaxis(stress_components, 0, -1).reshape(H, W, 3)

    return stress_map
