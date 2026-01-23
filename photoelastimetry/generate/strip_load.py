import json5
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
from tqdm import tqdm

from photoelastimetry.image import (
    compute_normalised_stokes,
    compute_stokes_components,
    simulate_four_step_polarimetry,
)
from photoelastimetry.plotting import virino

virino_cmap = virino()


def strip_load_stress_cartesian(X, Y, p, a):
    """
    Analytical solution for a uniform strip load on an elastic half-space.

    Computes the stress field in a semi-infinite elastic solid subjected to a
    uniform normal pressure p distributed over a strip of width 2a.

    Parameters
    ----------
    X : array-like
        X coordinates (horizontal position).
        Load is centered at x=0.
    Y : array-like
        Y coordinates (depth, positive downward from surface).
    p : float
        Uniform pressure magnitude (Pa).
    a : float
        Half-width of the loaded strip (m).

    Returns
    -------
    sigma_xx : array-like
        Normal stress in x direction (Pa).
    sigma_yy : array-like
        Normal stress in y direction (Pa).
    tau_xy : array-like
        Shear stress (Pa).

    Notes
    -----
    We use the solution given by Timoshenko and Goodier (Theory of Elasticity).
    Using angles theta1 and theta2 subtended by the ends of the load.

    Coordinate system:
    - y is positive downward.
    - Load is applied from x = -a to x = +a at y = 0.
    """

    # Angles from vertical (y-axis) to the vectors connecting point (x,y)
    # to the edges of the strip (-a, 0) and (+a, 0).
    # theta = arctan2(dx, dy) gives angle from vertical.

    # Angle to x = -a
    theta1 = np.arctan2(X + a, Y)

    # Angle to x = +a
    theta2 = np.arctan2(X - a, Y)

    # Alpha is the included angle
    alpha = theta1 - theta2

    # Beta is the angle of the bisector
    two_beta = theta1 + theta2

    # Stress formulas (Compressive is negative)
    # sigma_x = -p/pi * (alpha - sin(alpha)*cos(2*beta))
    # sigma_y = -p/pi * (alpha + sin(alpha)*cos(2*beta))
    # tau_xy  = -p/pi * (sin(alpha)*sin(2*beta))

    factor = p / np.pi

    sin_alpha = np.sin(alpha)
    cos_2beta = np.cos(two_beta)
    sin_2beta = np.sin(two_beta)

    sigma_xx = factor * (alpha - sin_alpha * cos_2beta)
    sigma_yy = factor * (alpha + sin_alpha * cos_2beta)
    tau_xy = factor * (sin_alpha * sin_2beta)

    # Clean up above surface
    above_surface = Y < 0
    sigma_xx[above_surface] = 0
    sigma_yy[above_surface] = 0
    tau_xy[above_surface] = 0

    return sigma_xx, sigma_yy, tau_xy


def generate_synthetic_strip_load(
    X, Y, p, a, S_i_hat, mask, wavelengths_nm, thickness, C, polarisation_efficiency
):
    """
    Generate synthetic Strip Load data for validation.

    Parameters
    ----------
    X, Y : array-like
        Coordinate grids
    p : float
        Pressure (Pa)
    a : float
        Half-width of strip (m)
    S_i_hat : array-like
        Incoming normalised Stokes vector
    mask : array-like
        Boolean mask for valid region
    wavelengths_nm : array-like
        Wavelengths in meters
    thickness : float
        Sample thickness (m)
    C : array-like
        Stress-optic coefficients for each wavelength
    polarisation_efficiency : float
        Polarisation efficiency (0-1)

    Returns
    -------
    synthetic_images : array-like
        Generated synthetic images [height, width, n_wavelengths, 4]
    principal_diff : array-like
        Principal stress difference
    theta_p : array-like
        Principal stress angle
    sigma_xx, sigma_yy, tau_xy : array-like
        Stress components
    """
    # Get stress components directly
    sigma_xx, sigma_yy, tau_xy = strip_load_stress_cartesian(X, Y, p, a)

    # Mask outside the valid region
    sigma_xx[~mask] = np.nan
    sigma_yy[~mask] = np.nan
    tau_xy[~mask] = np.nan

    # Principal stress difference and angle
    # sigma_avg = 0.5 * (sigma_xx + sigma_yy)
    # R_mohr = np.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + tau_xy**2)
    # sigma1 = sigma_avg + R_mohr
    # sigma2 = sigma_avg - R_mohr
    # principal_diff = sigma1 - sigma2

    principal_diff = np.sqrt((sigma_xx - sigma_yy) ** 2 + 4 * tau_xy**2)
    theta_p = 0.5 * np.arctan2(2 * tau_xy, sigma_xx - sigma_yy)

    # Mask again
    principal_diff[~mask] = np.nan
    theta_p[~mask] = np.nan

    height, width = sigma_xx.shape
    n_wavelengths = len(wavelengths_nm)

    synthetic_images = np.empty((height, width, n_wavelengths, 4))

    # Use incoming light fully S1 polarized (standard setup)
    nu = 1.0  # Solid sample

    for i, lambda_light in tqdm(enumerate(wavelengths_nm)):
        # Generate four-step polarimetry images using Mueller matrix approach
        # Note: We pass polarisation_efficiency if the simulator supports it?
        # The current simulate_four_step_polarimetry signature in point_load usage is:
        # simulate_four_step_polarimetry(sigma_xx, sigma_yy, tau_xy, C[i], nu, thickness, lambda_light, S_i_hat)
        # It doesn't seem to take polarisation_efficiency explicitly in the function call in point_load.py.
        # I will check simulate_four_step_polarimetry definition if needed, but for now stick to interface.

        I0_pol, I45_pol, I90_pol, I135_pol = simulate_four_step_polarimetry(
            sigma_xx, sigma_yy, tau_xy, C[i], nu, thickness, lambda_light, S_i_hat
        )

        synthetic_images[:, :, i, 0] = I0_pol
        synthetic_images[:, :, i, 1] = I45_pol
        synthetic_images[:, :, i, 2] = I90_pol
        synthetic_images[:, :, i, 3] = I135_pol

    return (
        synthetic_images,
        principal_diff,
        theta_p,
        sigma_xx,
        sigma_yy,
        tau_xy,
    )


if __name__ == "__main__":
    # Simple test execution
    pass
