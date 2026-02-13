"""
Image processing and photoelastic forward model functions.

This module contains helper functions for polarimetric image analysis and
photoelastic forward modeling, including stress-to-optical transformations.
"""

import numpy as np


def DoLP(image):
    """
    Calculate the Degree of Linear Polarisation (DoLP).
    """
    I = np.sum(image, axis=3)  # total intensity ovr all polarisation states

    Q = image[:, :, :, 0] - image[:, :, :, 1]  # 0/90 difference
    U = image[:, :, :, 2] - image[:, :, :, 3]  # 45/135 difference

    return np.sqrt(Q**2 + U**2) / I


def AoLP(image):
    """
    Calculate the Angle of Linear Polarisation (AoLP).
    """

    Q = image[:, :, :, 0] - image[:, :, :, 1]  # 0/90 difference
    U = image[:, :, :, 2] - image[:, :, :, 3]  # 45/135 difference

    return 0.5 * np.arctan2(U, Q)


def compute_retardance(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength):
    """
    Compute retardance for a given stress tensor and material properties.

    Parameters
    ----------
    sigma_xx : float or array-like
        Normal stress component in x direction (Pa).
    sigma_yy : float or array-like
        Normal stress component in y direction (Pa).
    sigma_xy : float or array-like
        Shear stress component (Pa).
    C : float
        Stress-optic coefficient for the colour channel (1/Pa).
    nu : float
        Solid fraction (dimensionless).
        For solid samples, use nu=1.0. For porous samples, this represents
        the effective optical path length factor relative to sample thickness.
    L : float
        Sample thickness (m).
    wavelength : float
        Wavelength of light (m).

    Returns
    -------
    delta : float or array-like
        Retardance (radians).

    Notes
    -----
    The retardance formula is: δ = (2πCnL/λ) * √[(σ_xx - σ_yy)² + 4σ_xy²]
    where the principal stress difference determines the birefringence magnitude.
    """
    principal_stress_diff = np.sqrt((sigma_xx - sigma_yy) ** 2 + 4 * sigma_xy**2)
    delta = (2 * np.pi * C * nu * L / wavelength) * principal_stress_diff
    return delta


def compute_principal_angle(sigma_xx, sigma_yy, sigma_xy):
    """
    Compute the orientation angle of the principal stress direction.

    Parameters
    ----------
    sigma_xx : float or array-like
        Normal stress component in x direction (Pa).
    sigma_yy : float or array-like
        Normal stress component in y direction (Pa).
    sigma_xy : float or array-like
        Shear stress component (Pa).

    Returns
    -------
    theta : float or array-like
        Principal stress orientation angle (radians).

    Notes
    -----
    In photoelasticity, the fast axis aligns with the maximum compressive
    stress direction. This formula gives the angle to the maximum tensile
    stress (σ_max).
    """
    theta = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)
    return theta


def mueller_matrix(theta, delta):
    """
    Compute the Mueller matrix for a birefringent material.

    Parameters
    ----------
    theta : float or array-like
        Orientation angle of principal stress direction (radians).
    delta : float or array-like
        Retardance (radians).

    Returns
    -------
    M : ndarray
        Mueller matrix (4x4) for scalar inputs, or (..., 4, 4) for array inputs.
    """
    cos_2theta = np.cos(2 * theta)
    sin_2theta = np.sin(2 * theta)
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    # Handle scalar vs array inputs
    if np.isscalar(theta) and np.isscalar(delta):
        M = np.array(
            [
                [1, 0, 0, 0],
                [
                    0,
                    cos_2theta**2 + sin_2theta**2 * cos_delta,
                    cos_2theta * sin_2theta * (1 - cos_delta),
                    sin_2theta * sin_delta,
                ],
                [
                    0,
                    cos_2theta * sin_2theta * (1 - cos_delta),
                    cos_2theta**2 * cos_delta + sin_2theta**2,
                    -cos_2theta * sin_delta,
                ],
                [0, -sin_2theta * sin_delta, cos_2theta * sin_delta, cos_delta],
            ]
        )
    else:
        # Array case - build matrix with proper shape (..., 4, 4)
        shape = np.broadcast(theta, delta).shape
        M = np.zeros(shape + (4, 4))

        M[..., 0, 0] = 1
        M[..., 1, 1] = cos_2theta**2 + sin_2theta**2 * cos_delta
        M[..., 1, 2] = cos_2theta * sin_2theta * (1 - cos_delta)
        M[..., 1, 3] = sin_2theta * sin_delta
        M[..., 2, 1] = cos_2theta * sin_2theta * (1 - cos_delta)
        M[..., 2, 2] = cos_2theta**2 * cos_delta + sin_2theta**2
        M[..., 2, 3] = -cos_2theta * sin_delta
        M[..., 3, 1] = -sin_2theta * sin_delta
        M[..., 3, 2] = cos_2theta * sin_delta
        M[..., 3, 3] = cos_delta

    return M


def mueller_matrix_sensitivity(theta, delta):
    """
    Compute derivatives of Mueller matrix elements with respect to theta and delta.

    Returns
    -------
    dM_dtheta : ndarray (..., 4, 4)
    dM_ddelta : ndarray (..., 4, 4)
    """
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    cd = np.cos(delta)
    sd = np.sin(delta)

    # Helper for 2*theta derivatives: d(c2)=-2s2, d(s2)=2c2
    dc2 = -2 * s2
    ds2 = 2 * c2

    shape = np.broadcast(theta, delta).shape
    dM_dtheta = np.zeros(shape + (4, 4))
    dM_ddelta = np.zeros(shape + (4, 4))

    # 0,0 is 1 -> derivs 0

    # 1,1: c2^2 + s2^2*cd
    dM_dtheta[..., 1, 1] = 2 * c2 * dc2 + 2 * s2 * ds2 * cd
    dM_ddelta[..., 1, 1] = -(s2**2) * sd

    # 1,2: c2*s2*(1-cd)
    dM_dtheta[..., 1, 2] = (dc2 * s2 + c2 * ds2) * (1 - cd)
    dM_ddelta[..., 1, 2] = c2 * s2 * sd

    # 1,3: s2*sd
    dM_dtheta[..., 1, 3] = ds2 * sd
    dM_ddelta[..., 1, 3] = s2 * cd

    # 2,1 = 1,2
    dM_dtheta[..., 2, 1] = dM_dtheta[..., 1, 2]
    dM_ddelta[..., 2, 1] = dM_ddelta[..., 1, 2]

    # 2,2: c2^2*cd + s2^2
    dM_dtheta[..., 2, 2] = (2 * c2 * dc2) * cd + 2 * s2 * ds2
    dM_ddelta[..., 2, 2] = -(c2**2) * sd

    # 2,3: -c2*sd
    dM_dtheta[..., 2, 3] = -dc2 * sd
    dM_ddelta[..., 2, 3] = -c2 * cd

    # 3,1: -s2*sd
    dM_dtheta[..., 3, 1] = -ds2 * sd
    dM_ddelta[..., 3, 1] = -s2 * cd

    # 3,2: c2*sd
    dM_dtheta[..., 3, 2] = dc2 * sd
    dM_ddelta[..., 3, 2] = c2 * cd

    # 3,3: cd
    dM_dtheta[..., 3, 3] = 0
    dM_ddelta[..., 3, 3] = -sd

    return dM_dtheta, dM_ddelta


def compute_stokes_components(I_0, I_45, I_90, I_135):
    """
    Compute the Stokes vector components (S0, S1, S2) from intensity measurements.

    Parameters
    ----------
    I_0 : array-like
        Intensity at polariser angle 0 degrees.
    I_45 : array-like
        Intensity at polariser angle 45 degrees.
    I_90 : array-like
        Intensity at polariser angle 90 degrees.
    I_135 : array-like
        Intensity at polariser angle 135 degrees.

    Returns
    -------
    S0 : array-like
        Total intensity (sum of orthogonal components).
    S1 : array-like
        Linear polarisation along 0-90 degrees.
    S2 : array-like
        Linear polarisation along 45-135 degrees.
    """
    S0 = I_0 + I_90
    S1 = I_0 - I_90
    S2 = I_45 - I_135
    return S0, S1, S2


def compute_normalised_stokes(S0, S1, S2):
    """
    Compute normalised Stokes vector components.

    Parameters
    ----------
    S0 : array-like
        Total intensity Stokes parameter.
    S1 : array-like
        First linear polarisation Stokes parameter.
    S2 : array-like
        Second linear polarisation Stokes parameter.

    Returns
    -------
    S1_hat : array-like
        Normalised S1 component (S1/S0).
    S2_hat : array-like
        Normalised S2 component (S2/S0).
    """
    S0_safe = np.where(S0 == 0, 1e-10, S0)
    S1_hat = S1 / S0_safe
    S2_hat = S2 / S0_safe
    return S1_hat, S2_hat


def predict_stokes(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength, S_i_hat):
    """
    Predict measured normalised Stokes components from a stress state.

    Parameters
    ----------
    sigma_xx, sigma_yy, sigma_xy : float or array-like
        Stress tensor components (Pa).
    C : float
        Stress-optic coefficient (1/Pa).
    nu : float
        Solid fraction.
    L : float
        Sample thickness (m).
    wavelength : float
        Illumination wavelength (m).
    S_i_hat : array-like
        Incoming normalised Stokes state [S1_hat, S2_hat] or [S1_hat, S2_hat, S3_hat].

    Returns
    -------
    ndarray
        Predicted measured [S1_hat, S2_hat].
    """
    theta = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)
    delta = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength)
    M = mueller_matrix(theta, delta)

    S_i_hat = np.asarray(S_i_hat)
    if len(S_i_hat) == 2:
        S_i_full = np.array([1.0, S_i_hat[0], S_i_hat[1], 0.0])
    elif len(S_i_hat) == 3:
        S_i_full = np.array([1.0, S_i_hat[0], S_i_hat[1], S_i_hat[2]])
    else:
        raise ValueError(f"S_i_hat must have length 2 or 3, got {len(S_i_hat)}")

    if M.ndim == 2:
        S_m = M @ S_i_full
        return S_m[1:3]

    S_m = np.einsum("...ij,j->...i", M, S_i_full)
    return S_m[..., 1:3]


def simulate_four_step_polarimetry(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength, S_i_hat, I0=1.0):
    """
    Simulate four-step polarimetry using Mueller matrix formalism.


    Parameters
    ----------
    sigma_xx : float or array-like
        Normal stress component in x direction (Pa).
    sigma_yy : float or array-like
        Normal stress component in y direction (Pa).
    sigma_xy : float or array-like
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
    I0 : float
        Incident light intensity (default: 1.0).

    Returns
    -------
    Four intensity images for analyzer angles 0°, 45°, 90°, 135°
    """
    # Compute retardance and principal angle from stress tensor
    theta = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)
    delta = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength)

    # Get Mueller matrix
    M = mueller_matrix(theta, delta)
    # Create full incoming Stokes vector
    S_i_hat = np.asarray(S_i_hat)
    if len(S_i_hat) == 2:
        # Backward compatibility: assume S3 = 0 (no circular polarisation)
        S_i_full = np.array([1.0, S_i_hat[0], S_i_hat[1], 0.0])
    elif len(S_i_hat) == 3:
        # Use provided circular component
        S_i_full = np.array([1.0, S_i_hat[0], S_i_hat[1], S_i_hat[2]])
    else:
        raise ValueError(f"S_i_hat must have 2 or 3 elements, got {len(S_i_hat)}")

    # Apply Mueller matrix to get output Stokes vector
    if M.ndim == 2:
        # Single pixel case
        S_out = M @ S_i_full
    else:
        # Array case - need to handle broadcasting
        S_out = np.einsum("...ij,j->...i", M, S_i_full)

    # Extract S0, S1, S2 from output
    S0_out = S_out[..., 0] if S_out.ndim > 1 else S_out[0]
    S1_out = S_out[..., 1] if S_out.ndim > 1 else S_out[1]
    S2_out = S_out[..., 2] if S_out.ndim > 1 else S_out[2]

    # Compute intensities for four analyzer angles
    # I(α) = (S0 + S1*cos(2α) + S2*sin(2α)) / 2
    I0_pol = I0 * (S0_out + S1_out) / 2  # α = 0°
    I45_pol = I0 * (S0_out + S2_out) / 2  # α = 45°
    I90_pol = I0 * (S0_out - S1_out) / 2  # α = 90°
    I135_pol = I0 * (S0_out - S2_out) / 2  # α = 135°

    return I0_pol, I45_pol, I90_pol, I135_pol
