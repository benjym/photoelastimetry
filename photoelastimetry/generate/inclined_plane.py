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


def inclined_stress_cartesian(X, Y, rho, g=9.81, theta_deg=0.0, K0=0.5):
    """
    Stress field for an inclined plane (rectangular mass with inclined gravity).

    This computes the stress field in a rectangular domain with gravity
    acting at an angle theta from vertical. The resulting stress field
    includes both normal stresses that vary with depth and shear stresses
    due to the inclined gravity.

    Parameters
    ----------
    X : array-like
        X coordinates (horizontal position).
    Y : array-like
        Y coordinates (depth, positive downward from surface).
    rho : float
        Density (kg/m^3).
    g : float
        Gravitational acceleration (m/s^2).
    theta_deg : float
        Inclination angle of gravity from vertical (degrees).
        0 degrees = vertical (standard lithostatic)
        Positive angles tilt gravity towards +x direction
    K0 : float
        Coefficient of lateral earth pressure at rest.
        Used to relate stresses perpendicular to the inclined direction.

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
    For an inclined gravity field, the body forces are:
    - f_x = rho * g * sin(theta)  (horizontal component)
    - f_y = rho * g * cos(theta)  (vertical component)

    The stress field must satisfy equilibrium:
    - dσ_xx/dx + dτ_xy/dy + f_x = 0
    - dτ_xy/dx + dσ_yy/dy + f_y = 0

    For a uniform rectangular mass with no boundaries except at the top,
    we assume a solution where stresses increase linearly with depth
    in the direction of gravity, and include shear stress from the
    inclined component.
    """
    theta_rad = np.deg2rad(theta_deg)

    # Components of gravity
    g_x = g * np.sin(theta_rad)  # Horizontal component
    g_y = g * np.cos(theta_rad)  # Vertical component

    # For a simple inclined plane solution, we assume:
    # 1. Vertical stress component increases with depth due to g_y
    # 2. Horizontal stress is related by K0
    # 3. Shear stress arises from the horizontal gravity component

    # Stress in the direction of gravity (normal to inclined layers)
    # For simplicity, we compute stress as if in a rotated coordinate system
    # then transform back to x-y coordinates

    # Normal stress perpendicular to gravity direction
    sigma_normal = rho * g * Y  # Total weight effect with depth

    # In the inclined case, we decompose this into x-y components
    # The vertical stress component
    sigma_yy = rho * g_y * Y

    # The horizontal stress has contributions from both K0 effect and inclination
    sigma_xx = K0 * sigma_yy

    # Shear stress arises from the inclined gravity
    # In equilibrium, tau_xy must balance the horizontal body force
    # For a linear variation: dtau_xy/dy = -rho * g_x
    # Integrating: tau_xy = -rho * g_x * Y + C
    # With free surface at Y=0: tau_xy(Y=0) = 0, so C = 0
    tau_xy = rho * g_x * Y

    # Ensure non-negative normal stresses (no tension)
    sigma_yy = np.maximum(0, sigma_yy)
    sigma_xx = np.maximum(0, sigma_xx)

    return sigma_xx, sigma_yy, tau_xy


def generate_synthetic_inclined_plane(X, Y, rho, g, theta_deg, K0, S_i_hat, wavelengths_nm, thickness, C):
    """
    Generate synthetic inclined plane stress data.

    Parameters
    ----------
    X, Y : array-like
        Coordinate grids
    rho : float
        Density (kg/m^3)
    g : float
        Gravitational acceleration (m/s^2)
    theta_deg : float
        Inclination angle from vertical (degrees)
    K0 : float
        Coefficient of lateral earth pressure
    S_i_hat : array-like
        Incoming normalised Stokes vector
    wavelengths_nm : array-like
        Wavelengths in meters
    thickness : float
        Sample thickness (m)
    C : array-like
        Stress-optic coefficients for each wavelength

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
    # Get stress components
    sigma_xx, sigma_yy, tau_xy = inclined_stress_cartesian(X, Y, rho, g, theta_deg, K0)

    # Principal stress difference and angle
    sigma_avg = 0.5 * (sigma_xx + sigma_yy)
    R_mohr = np.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + tau_xy**2)
    sigma1 = sigma_avg + R_mohr
    sigma2 = sigma_avg - R_mohr
    principal_diff = sigma1 - sigma2
    theta_p = 0.5 * np.arctan2(2 * tau_xy, sigma_xx - sigma_yy)

    height, width = sigma_xx.shape
    n_colors = len(wavelengths_nm)

    synthetic_images = np.zeros((height, width, n_colors, 4))

    for i, lambda_light in enumerate(wavelengths_nm):
        # Stress-optic coefficient for this wavelength
        c_lambda = C[i]

        # Simulate polarimetry
        I0, I45, I90, I135 = simulate_four_step_polarimetry(
            sigma_xx, sigma_yy, tau_xy, c_lambda, 1.0, thickness, lambda_light, S_i_hat  # nu (solid fraction)
        )

        # Stack into our array
        synthetic_images[:, :, i, 0] = I0
        synthetic_images[:, :, i, 1] = I45
        synthetic_images[:, :, i, 2] = I90
        synthetic_images[:, :, i, 3] = I135

    return synthetic_images, principal_diff, theta_p, sigma_xx, sigma_yy, tau_xy


if __name__ == "__main__":
    import os

    import photoelastimetry.io

    # Parameters for inclined plane
    width_m = 0.1  # 10 cm width
    height_m = 0.1  # 10 cm depth
    rho = 2500  # Density kg/m3 (rock-like)
    g = 9.81  # Gravity
    theta_deg = 30.0  # Inclination angle (30 degrees from vertical)
    K0 = 0.5  # Horizontal stress is half vertical stress

    polarisation_angle_deg = 10.0  # Incoming polarisation angle
    polarisation_angle_rad = np.deg2rad(polarisation_angle_deg)
    S_i_hat = np.array([np.cos(2 * polarisation_angle_rad), np.sin(2 * polarisation_angle_rad)])

    with open("json/test.json5", "r") as f:
        params = json5.load(f)

    thickness = params["thickness"]
    wavelengths_nm = np.array(params["wavelengths"]) * 1e-9
    C = np.array(params["C"]) * 50

    # Grid
    n_res = 512
    x = np.linspace(0, width_m, n_res)
    y = np.linspace(0, height_m, n_res)  # y=0 at top
    X, Y = np.meshgrid(x, y)

    # Generate
    synthetic_images, principal_diff, theta_p, sigma_xx, sigma_yy, tau_xy = generate_synthetic_inclined_plane(
        X, Y, rho, g, theta_deg, K0, S_i_hat, wavelengths_nm, thickness, C
    )

    # Save
    stress = np.stack((sigma_xx, sigma_yy, tau_xy), axis=-1)

    os.makedirs("images/inclined_plane", exist_ok=True)

    photoelastimetry.io.save_image("images/inclined_plane/inclined_plane_stress.tiff", stress)
    photoelastimetry.io.save_image("images/inclined_plane/inclined_plane_images.tiff", synthetic_images)

    # Plot True Stress Components
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")

    # Sigma XX
    im = axes[0, 0].imshow(sigma_xx / 1e6, extent=[0, width_m, height_m, 0], cmap="viridis")
    plt.colorbar(im, ax=axes[0, 0], label="σ_xx (MPa)")
    axes[0, 0].set_title(f"Horizontal Stress σ_xx (θ={theta_deg}°)")
    axes[0, 0].set_xlabel("Width (m)")
    axes[0, 0].set_ylabel("Depth (m)")

    # Sigma YY
    im = axes[0, 1].imshow(sigma_yy / 1e6, extent=[0, width_m, height_m, 0], cmap="viridis")
    plt.colorbar(im, ax=axes[0, 1], label="σ_yy (MPa)")
    axes[0, 1].set_title(f"Vertical Stress σ_yy (θ={theta_deg}°)")
    axes[0, 1].set_xlabel("Width (m)")
    axes[0, 1].set_ylabel("Depth (m)")

    # Tau XY
    im = axes[0, 2].imshow(tau_xy / 1e6, extent=[0, width_m, height_m, 0], cmap="RdBu_r")
    plt.colorbar(im, ax=axes[0, 2], label="τ_xy (MPa)")
    axes[0, 2].set_title(f"Shear Stress τ_xy (θ={theta_deg}°)")
    axes[0, 2].set_xlabel("Width (m)")
    axes[0, 2].set_ylabel("Depth (m)")

    # Principal Stress Difference
    im = axes[1, 0].imshow(principal_diff / 1e6, extent=[0, width_m, height_m, 0], cmap="plasma")
    plt.colorbar(im, ax=axes[1, 0], label="σ₁ - σ₂ (MPa)")
    axes[1, 0].set_title("Principal Stress Difference")
    axes[1, 0].set_xlabel("Width (m)")
    axes[1, 0].set_ylabel("Depth (m)")

    # Principal Angle
    im = axes[1, 1].imshow(
        np.rad2deg(theta_p), extent=[0, width_m, height_m, 0], cmap=virino_cmap, vmin=-90, vmax=90
    )
    plt.colorbar(im, ax=axes[1, 1], label="θ_p (degrees)")
    axes[1, 1].set_title("Principal Stress Angle")
    axes[1, 1].set_xlabel("Width (m)")
    axes[1, 1].set_ylabel("Depth (m)")

    # Sample polarimetry image (first wavelength, 0° polarizer)
    im = axes[1, 2].imshow(synthetic_images[:, :, 0, 0], extent=[0, width_m, height_m, 0], cmap="gray")
    plt.colorbar(im, ax=axes[1, 2], label="Intensity")
    axes[1, 2].set_title("Sample Polarimetry Image (0°)")
    axes[1, 2].set_xlabel("Width (m)")
    axes[1, 2].set_ylabel("Depth (m)")

    plt.savefig("images/inclined_plane/inclined_plane_visualization.png", dpi=150)

    print(f"Generated inclined plane stress dataset with θ={theta_deg}°.")
    print(f"Max σ_yy: {np.max(sigma_yy)/1e3:.2f} kPa")
    print(f"Max σ_xx: {np.max(sigma_xx)/1e3:.2f} kPa")
    print(f"Max τ_xy: {np.max(np.abs(tau_xy))/1e3:.2f} kPa")
