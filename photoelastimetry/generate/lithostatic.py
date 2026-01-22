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


def lithostatic_stress_cartesian(X, Y, rho, g=9.81, K0=0.5):
    """
    Lithostatic stress field (increasing with depth).

    Parameters
    ----------
    X : array-like
        X coordinates (horizontal position).
    Y : array-like
        Y coordinates (depth, positive downward from surface).
    rho : float
        Density (kg/m^3).
    g : float
        Gravity (m/s^2).
    K0 : float
        Coefficient of lateral earth pressure (sigma_xx / sigma_yy).

    Returns
    -------
    sigma_xx : array-like
        Normal stress in x direction (Pa).
    sigma_yy : array-like
        Normal stress in y direction (Pa).
    tau_xy : array-like
        Shear stress (Pa).
    """
    # Vertical stress increases linearly with depth
    sigma_yy = rho * g * Y

    # Ensure no negative depth (if Y coordinate system differs) produces negative stress
    # Assuming soil/rock/granular material doesn't support tension this way usually
    sigma_yy = np.maximum(0, sigma_yy)

    # Horizontal stress is a fraction of vertical stress
    sigma_xx = K0 * sigma_yy

    # No shear stress in simple lithostatic case
    tau_xy = np.zeros_like(X)

    return sigma_xx, sigma_yy, tau_xy


def generate_synthetic_lithostatic(X, Y, rho, g, K0, S_i_hat, wavelengths_nm, thickness, C):
    """
    Generate synthetic lithostatic stress data.
    """

    # Get stress components
    sigma_xx, sigma_yy, tau_xy = lithostatic_stress_cartesian(X, Y, rho, g, K0)

    # Principal stress difference and angle
    principal_diff = np.abs(sigma_xx - sigma_yy)
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

    # Parameters
    width_m = 0.1  # 10 cm width
    height_m = 0.1  # 10 cm depth
    rho = 2500  # Density kg/m3 (rock-like)
    g = 9.81  # Gravity
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
    synthetic_images, principal_diff, theta_p, sigma_xx, sigma_yy, tau_xy = generate_synthetic_lithostatic(
        X, Y, rho, g, K0, S_i_hat, wavelengths_nm, thickness, C
    )

    # Save
    stress = np.stack((sigma_xx, sigma_yy, tau_xy), axis=-1)
    # stress = np.nan_to_num(stress, nan=0.0) # Not needed here

    os.makedirs("images/lithostatic", exist_ok=True)

    photoelastimetry.io.save_image("images/lithostatic/lithostatic_stress.tiff", stress)
    photoelastimetry.io.save_image("images/lithostatic/lithostatic_images.tiff", synthetic_images)

    # Plot True Stress Difference
    fig = plt.figure(figsize=(6, 5), layout="constrained")
    im = plt.imshow(principal_diff, extent=[0, width_m, height_m, 0], cmap="viridis")  # y increases down
    plt.colorbar(im, label="Principal Stress Difference (Pa)")
    plt.title(f"Lithostatic Stress Diff (K0={K0})")
    plt.xlabel("Width (m)")
    plt.ylabel("Depth (m)")
    plt.savefig("images/lithostatic/true_stress_diff.png")

    print("Generated lithostatic stress dataset.")
