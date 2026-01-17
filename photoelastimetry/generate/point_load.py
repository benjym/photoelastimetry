import json5
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
from tqdm import tqdm

from photoelastimetry.image import (
    compute_normalized_stokes,
    compute_stokes_components,
    simulate_four_step_polarimetry,
)
from photoelastimetry.plotting import virino

virino_cmap = virino()


def boussinesq_stress_cartesian(X, Y, P, nu_poisson=0.3):
    """
    Boussinesq solution for point load on elastic half-space.

    This computes the stress field in a semi-infinite elastic solid
    subjected to a concentrated vertical point load P at the surface.

    Parameters
    ----------
    X : array-like
        X coordinates (horizontal position)
    Y : array-like
        Y coordinates (depth, positive downward from surface)
    P : float
        Point load magnitude (force, in N)
    nu_poisson : float
        Poisson's ratio (default: 0.3)

    Returns
    -------
    sigma_xx : array-like
        Normal stress in x direction (Pa)
    sigma_yy : array-like
        Normal stress in y direction (Pa)
    tau_xy : array-like
        Shear stress (Pa)

    Notes
    -----
    The Boussinesq solution assumes:
    - Semi-infinite elastic half-space (y >= 0)
    - Point load P applied at origin on the surface
    - Y axis points downward (into the material)
    - Linear elastic material with Poisson's ratio nu

    References
    ----------
    Boussinesq, J. (1885). Application des potentiels à l'étude de
    l'équilibre et du mouvement des solides élastiques.
    """
    X_safe = X.copy()
    Y_safe = Y.copy()

    # Small offset to avoid singularity at load point
    epsilon = 1e-6
    origin_mask = (X**2 + Y**2) < epsilon**2
    X_safe = np.where(origin_mask, epsilon, X_safe)
    Y_safe = np.where(origin_mask, epsilon, Y_safe)

    # Distance from load point
    r = np.sqrt(X_safe**2 + Y_safe**2)

    # Boussinesq stress components for point load
    # These are the classical solutions from elasticity theory
    r3 = r**3
    r5 = r**5

    # Stress components
    sigma_xx = -(P / (2 * np.pi)) * (
        (1 - 2 * nu_poisson) * (Y_safe / r3 - X_safe**2 * Y_safe / r5) - 3 * X_safe**2 * Y_safe / r5
    )

    sigma_yy = -(P / (2 * np.pi)) * (
        (1 - 2 * nu_poisson) * (Y_safe / r3 - Y_safe**3 / r5) - 3 * Y_safe**3 / r5
    )

    tau_xy = -(P / (2 * np.pi)) * (
        (1 - 2 * nu_poisson) * (-X_safe / r3 + X_safe * Y_safe**2 / r5) - 3 * X_safe * Y_safe**2 / r5
    )

    # Set stress to zero above the surface (y < 0)
    above_surface = Y < 0
    sigma_xx[above_surface] = 0
    sigma_yy[above_surface] = 0
    tau_xy[above_surface] = 0

    return sigma_xx, sigma_yy, tau_xy


def generate_synthetic_boussinesq(
    X, Y, P, nu_poisson, S_i_hat, mask, wavelengths_nm, thickness, C, polarisation_efficiency
):
    """
    Generate synthetic Boussinesq point load data for validation.

    This function creates a synthetic dataset based on the analytical solution
    and saves it in a format suitable for testing.

    Parameters
    ----------
    X, Y : array-like
        Coordinate grids
    P : float
        Point load magnitude (N)
    nu_poisson : float
        Poisson's ratio
    S_i_hat : array-like
        Incoming normalized Stokes vector
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
    sigma_xx, sigma_yy, tau_xy = boussinesq_stress_cartesian(X, Y, P, nu_poisson)

    # Mask outside the valid region
    sigma_xx[~mask] = np.nan
    sigma_yy[~mask] = np.nan
    tau_xy[~mask] = np.nan

    # Principal stress difference and angle
    sigma_avg = 0.5 * (sigma_xx + sigma_yy)
    R_mohr = np.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + tau_xy**2)
    sigma1 = sigma_avg + R_mohr
    sigma2 = sigma_avg - R_mohr
    principal_diff = sigma1 - sigma2
    theta_p = 0.5 * np.arctan2(2 * tau_xy, sigma_xx - sigma_yy)

    # Mask again
    principal_diff[~mask] = np.nan
    theta_p[~mask] = np.nan

    height, width = sigma_xx.shape
    n_wavelengths = len(wavelengths_nm)

    synthetic_images = np.empty((height, width, n_wavelengths, 4))  # wavelengths, 4 polarizer angles

    # Use incoming light fully S1 polarized (standard setup)
    nu = 1.0  # Solid sample

    for i, lambda_light in tqdm(enumerate(wavelengths_nm)):
        # Generate four-step polarimetry images using Mueller matrix approach
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


def post_process_synthetic_data(
    X, Y, principal_diff, theta_p, sigma_xx, sigma_yy, tau_xy, S_i_hat, t_sample, C, lambda_light, P, outname
):
    """
    Post-process and visualize synthetic Boussinesq data.

    Parameters
    ----------
    X, Y : array-like
        Coordinate grids
    principal_diff : array-like
        Principal stress difference
    theta_p : array-like
        Principal stress angle
    sigma_xx, sigma_yy, tau_xy : array-like
        Stress components
    S_i_hat : array-like
        Incoming normalized Stokes vector
    t_sample : float
        Sample thickness (m)
    C : float
        Stress-optic coefficient (1/Pa)
    lambda_light : float
        Wavelength (m)
    P : float
        Point load (N)
    outname : str
        Output filename
    """
    plt.figure(figsize=(12, 12), layout="constrained")

    # Calculate retardation
    retardation = (2 * np.pi * t_sample * C * principal_diff) / lambda_light
    f_sigma = lambda_light / (2 * C * t_sample)  # material fringe value
    fringe_order = principal_diff / f_sigma  # N = (σ1 - σ2)/f_σ

    # Photoelastic parameters
    # For circular polariscope (dark field): I ∝ sin²(δ/2) where δ is retardation
    intensity_dark = np.sin(retardation / 2) ** 2  # Dark field intensity

    # For isoclinic lines, we need the extinction angle in plane polariscope
    isoclinic_angle = theta_p  # Principal stress angle

    # Generate four-step polarimetry images using Mueller matrix approach
    nu = 1.0  # Solid sample
    I0_pol, I45_pol, I90_pol, I135_pol = simulate_four_step_polarimetry(
        sigma_xx, sigma_yy, tau_xy, C, nu, t_sample, lambda_light, S_i_hat
    )

    # Calculate Stokes parameters from polarimetry
    S0, S1, S2 = compute_stokes_components(I0_pol, I45_pol, I90_pol, I135_pol)
    S1_hat, S2_hat = compute_normalized_stokes(S0, S1, S2)

    # Degree of linear polarization
    DoLP = np.sqrt(S1_hat**2 + S2_hat**2)

    # Angle of linear polarization
    AoLP = np.mod(0.5 * np.arctan2(S2_hat, S1_hat), np.pi)

    # Plot characteristic photoelastic patterns
    plt.clf()

    plt.subplot(4, 4, 1)
    # Plot fringe order
    max_fringe = np.nanmax(fringe_order)
    if max_fringe > 0:
        levels = np.linspace(0, min(max_fringe, 8), 25)
        plt.contourf(X, Y, fringe_order, levels=levels, cmap="plasma", extend="max")
        plt.colorbar(label="Fringe Order N", shrink=0.8)
        # Add integer fringe contour lines (dark fringes)
        integer_levels = np.arange(0.5, min(max_fringe, 8), 1.0)
        plt.contour(
            X,
            Y,
            fringe_order,
            levels=integer_levels,
            colors="black",
            linewidths=1.0,
        )
    plt.title("Isochromatic Fringes")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 2)
    # Dark field circular polariscope
    plt.contourf(X, Y, intensity_dark, levels=50, cmap="gray")
    plt.colorbar(label="Intensity", shrink=0.8)
    plt.title("Dark Field Circular\nPolariscope")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 3)
    # Principal stress directions (isoclinics)
    isoclinic_angle_deg = np.rad2deg(isoclinic_angle)
    # Wrap to [-90, 90] for better visualization
    isoclinic_angle_deg = ((isoclinic_angle_deg + 90) % 180) - 90
    plt.contourf(X, Y, isoclinic_angle_deg, levels=36, cmap=virino_cmap)
    plt.colorbar(label="Isoclinic Angle (°)", shrink=0.8)
    plt.title("Isoclinic Lines\n(Principal Stress Direction)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 4)
    plt.contourf(X, Y, DoLP, cmap="viridis")
    plt.colorbar(label="DoLP", shrink=0.8)
    plt.title("Degree of Linear\nPolarization")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 5)
    plt.contourf(X, Y, AoLP, levels=36, cmap=virino_cmap, vmin=0, vmax=np.pi)
    plt.colorbar(label="AoLP (rad)", shrink=0.8)
    plt.title("Angle of Linear\nPolarization")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    # Second row: Four-step polarimetry images
    polarizer_angles = ["0°", "45°", "90°", "135°"]
    polarimetry_images = [I0_pol, I45_pol, I90_pol, I135_pol]

    for i, (img, angle) in enumerate(zip(polarimetry_images, polarizer_angles)):
        plt.subplot(4, 4, 6 + i)
        plt.contourf(X, Y, img, levels=50, cmap="gray")
        plt.colorbar(label="Intensity", shrink=0.8)
        plt.title(f"Linear Polarizer at {angle}")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.gca().set_aspect("equal")

    # Add intensity range plot
    plt.subplot(4, 4, 10)
    intensity_range = np.maximum.reduce(polarimetry_images) - np.minimum.reduce(polarimetry_images)
    plt.contourf(X, Y, intensity_range, levels=50, cmap="hot")
    plt.colorbar(label="Intensity Range", shrink=0.8)
    plt.title("Polarimetric Contrast\n(Max - Min Intensity)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    # Third row: Stress components
    plt.subplot(4, 4, 11)
    sigma_xx_MPa = sigma_xx / 1e6  # Convert to MPa
    sigma_xx_max = np.nanmax(np.abs(sigma_xx_MPa))
    if sigma_xx_max > 0:
        plt.pcolormesh(
            X,
            Y,
            sigma_xx_MPa,
            cmap="RdBu_r",
            norm=SymLogNorm(linthresh=sigma_xx_max / 1e3, vmin=-sigma_xx_max, vmax=sigma_xx_max),
        )
    plt.colorbar(label="σ_xx (MPa)", shrink=0.8)
    plt.title("Horizontal Stress σ_xx")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 12)
    sigma_yy_MPa = sigma_yy / 1e6
    sigma_yy_max = np.nanmax(np.abs(sigma_yy_MPa))
    if sigma_yy_max > 0:
        plt.pcolormesh(
            X,
            Y,
            sigma_yy_MPa,
            cmap="RdBu_r",
            norm=SymLogNorm(
                linthresh=sigma_yy_max / 1e3,
                vmin=-sigma_yy_max,
                vmax=sigma_yy_max,
            ),
        )
    plt.colorbar(label="σ_yy (MPa)", shrink=0.8)
    plt.title("Vertical Stress σ_yy")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 13)
    tau_xy_MPa = tau_xy / 1e6
    tau_xy_max = np.nanmax(np.abs(tau_xy_MPa))
    if tau_xy_max > 0:
        plt.pcolormesh(
            X,
            Y,
            tau_xy_MPa,
            cmap="RdBu_r",
            norm=SymLogNorm(linthresh=tau_xy_max / 1e3, vmin=-tau_xy_max, vmax=tau_xy_max),
        )
    plt.colorbar(label="τ_xy (MPa)", shrink=0.8)
    plt.title("Shear Stress τ_xy")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 14)
    principal_diff_MPa = principal_diff / 1e6  # Convert to MPa
    max_diff = np.nanmax(np.abs(principal_diff_MPa))
    if max_diff > 0:
        plt.pcolormesh(
            X,
            Y,
            principal_diff_MPa,
            cmap="plasma",
            norm=LogNorm(vmax=max_diff, vmin=1e-4 * max_diff),
        )
    plt.colorbar(label="σ₁ - σ₂ (MPa)", shrink=0.8)
    plt.title("Principal Stress\nDifference")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    plt.subplot(4, 4, 15)
    max_retardation = np.nanmax(np.abs(retardation))
    if max_retardation > 0:
        plt.pcolormesh(
            X,
            Y,
            retardation,
            cmap="plasma",
            norm=LogNorm(vmin=1e-4 * max_retardation, vmax=max_retardation),
        )
    plt.colorbar(label="Retardation", shrink=0.8)
    plt.title("Retardation")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect("equal")

    # Summary statistics
    plt.subplot(4, 4, 16)
    n = X.shape[0]
    center_x, center_y = n // 2, n // 2
    plt.text(
        0.1,
        0.8,
        f"Load: {P:.0f} N",
        fontsize=12,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.7,
        f"Max Fringe Order: {max_fringe:.2f}",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.6,
        f"Max σ₁-σ₂: {max_diff:.2f} MPa",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.5,
        f"Center σₓₓ: {sigma_xx[center_y, center_x]/1e6:.2f} MPa",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.4,
        f"Center σᵧᵧ: {sigma_yy[center_y, center_x]/1e6:.2f} MPa",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.3,
        f"Material f_σ: {f_sigma/1e6:.1f} MPa",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.2,
        f"Thickness: {t_sample*1000:.0f} mm",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.1,
        f"Wavelength: {lambda_light*1e9:.0f} nm",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.title("Experiment\nParameters")
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.gca().axis("off")

    plt.savefig(outname)


if __name__ == "__main__":
    import os

    import photoelastimetry.io

    plt.figure(figsize=(12, 12), layout="constrained")

    # Boussinesq point load parameters
    P = 1.0  # Point load (N)
    nu_poisson = 0.3  # Poisson's ratio for elastic material

    with open("json/elastic.json5", "r") as f:
        params = json5.load(f)

    thickness = params["thickness"]  # Thickness in m
    wavelengths_nm = np.array(params["wavelengths"]) * 1e-9  # Wavelengths in m
    C = np.array(params["C"])  # Stress-optic coefficient (Pa^-1) for each wavelength
    polarisation_efficiency = params["polarisation_efficiency"]  # Polarisation efficiency (0-1)

    # Grid for elastic half-space
    # Domain: x from -L to L, y from 0 to depth
    n = 100
    L = 0.02  # Half-width of domain (m)
    depth = 0.02  # Depth of domain (m)

    x = np.linspace(-L, L, n)
    y = np.linspace(0, depth, n)
    X, Y = np.meshgrid(x, y)

    # Mask for valid region (below surface)
    mask = Y >= 0

    # Get S_i_hat from params if available, otherwise use default
    S_i_hat = np.array(params.get("S_i_hat", [1.0, 0.0, 0.0]))

    # Generate synthetic Boussinesq data
    synthetic_images, principal_diff, theta_p, sigma_xx, sigma_yy, tau_xy = generate_synthetic_boussinesq(
        X,
        Y,
        P,
        nu_poisson,
        S_i_hat,
        mask,
        wavelengths_nm,
        thickness,
        C,
        polarisation_efficiency,
    )

    # Save the output data
    stress = np.stack((sigma_xx, sigma_yy, tau_xy), axis=-1)
    # remove nans
    stress = np.nan_to_num(stress, nan=0.0)

    os.makedirs("images/elastic", exist_ok=True)

    photoelastimetry.io.save_image("images/elastic/boussinesq_synthetic_stress.tiff", stress)
    photoelastimetry.io.save_image("images/elastic/boussinesq_synthetic_images.tiff", synthetic_images)

    plt.figure(figsize=(6, 4), layout="constrained")
    plt.imshow(principal_diff, norm=LogNorm())
    plt.colorbar(label="Principal Stress Difference (Pa)", orientation="vertical")
    plt.savefig("boussinesq_stress_difference.png")

    # Post-process and visualize the synthetic data
    for i, lambda_light in enumerate(wavelengths_nm):
        post_process_synthetic_data(
            X,
            Y,
            principal_diff,
            theta_p,
            sigma_xx,
            sigma_yy,
            tau_xy,
            S_i_hat,
            thickness,
            C[i],
            lambda_light,
            P,
            f"boussinesq_post_processed_{P:07.0f}_{i:02d}.png",
        )
