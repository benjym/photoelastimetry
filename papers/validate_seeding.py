import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from plot_config import configure_plots
from tqdm import tqdm

from photoelastimetry.image import compute_normalized_stokes, compute_stokes_components, mueller_matrix
from photoelastimetry.seeding import invert_wrapped_retardance, resolve_fringe_orders

configure_plots()

# This file makes a single figure that validates the seeding procedure for phase unwrapping.

# Material and optical properties
wavelengths = np.array([650e-9, 532e-9, 473e-9])  # R, G, B wavelengths (m)
C_values = np.array([2.3e-10, 2.5e-10, 2.7e-10])  # Stress-optic coefficients (1/Pa)
nu = 1.0  # Solid fraction
L = 0.01  # Sample thickness (m)
S_i_hat = np.array([1, 0])  # Incoming linearly polarized light at 0 degrees
theta_test = 1  # Test angle (radians)
n_max = 6  # max fringe order to search
sigma_max = n_max * wavelengths.min() / (C_values.max() * nu * L)
n_tests = 200
resolution = 20

plt.figure()

# debug = True
debug = False


def forward_project(theta, delta, S_i_full, I0=1.0):
    # Get Mueller matrix
    M = mueller_matrix(theta, delta)

    # Apply Mueller matrix to get output Stokes vector
    S_out = M @ S_i_full

    # Compute intensities for four analyzer angles
    # I(α) = (S0 + S1*cos(2α) + S2*sin(2α)) / 2
    I0_pol = I0 * (S_out[0] + S_out[1]) / 2  # α = 0°
    I45_pol = I0 * (S_out[0] + S_out[2]) / 2  # α = 45°
    I90_pol = I0 * (S_out[0] - S_out[1]) / 2  # α = 90°
    I135_pol = I0 * (S_out[0] - S_out[2]) / 2  # α = 135°

    return I0_pol, I45_pol, I90_pol, I135_pol


def test_seeding(
    theta, delta_sigma, wavelengths, C_values, nu, L, S_i_hat, sigma_max, n_max, SnR, debug=True
):
    retardation_true = 2 * np.pi * C_values * nu * L * delta_sigma / wavelengths
    delta_true = retardation_true % (2 * np.pi)

    # Generate "measured" Stokes components by forward prediction
    S_m_hat = np.zeros((3, 2))
    for c in range(3):
        I0_pol, I45_pol, I90_pol, I135_pol = forward_project(
            theta, delta_true[c], np.array([1.0, S_i_hat[0], S_i_hat[1], 0.0])
        )

        # Simulate Shot Noise (Poisson)
        # SnR provided is treated as the Peak SNR (at Intensity = 1.0)
        # Peak SNR = sqrt(N_max_photons)  => N_max = SnR^2
        # This models the discrete arrival of photons.

        n_photons_max = SnR**2  # Full well capacity in photons

        # Avoid zero or negative intensities for Poisson
        I0_pol = np.maximum(I0_pol, 0)
        I45_pol = np.maximum(I45_pol, 0)
        I90_pol = np.maximum(I90_pol, 0)
        I135_pol = np.maximum(I135_pol, 0)

        # Apply Poisson noise: I_noisy = Poisson(I * N_max) / N_max
        # We use a small epsilon to avoid issues if SnR is extremely low
        if n_photons_max > 1e-6:
            I0_pol = np.random.poisson(I0_pol * n_photons_max) / n_photons_max
            I45_pol = np.random.poisson(I45_pol * n_photons_max) / n_photons_max
            I90_pol = np.random.poisson(I90_pol * n_photons_max) / n_photons_max
            I135_pol = np.random.poisson(I135_pol * n_photons_max) / n_photons_max

        S = compute_stokes_components(I0_pol, I45_pol, I90_pol, I135_pol)

        this_normalised_stokes = compute_normalized_stokes(S[0], S[1], S[2])
        # print(this_normalised_stokes)
        S_m_hat[c, :] = np.array(this_normalised_stokes)

    # Use seeding procedure to estimate initial stress tensor
    theta, delta_wrap = invert_wrapped_retardance(S_m_hat)
    delta_sigma_guess = resolve_fringe_orders(
        delta_wrap, theta, wavelengths, C_values, nu, L, sigma_max, n_max
    )

    return np.abs(delta_sigma_guess - delta_sigma) / delta_sigma


plt.subplot(211)
# show all 12 I(retardance) curves for R, G, B and 4 analyzer angles
delta_sigma = np.linspace(0, sigma_max, 1000)
for c in range(3):
    I0_vals = []
    I45_vals = []
    I90_vals = []
    I135_vals = []
    retardance_vals = delta_sigma * 2 * np.pi * C_values[c] * nu * L / wavelengths[c]
    for delta in retardance_vals:
        I0_pol, I45_pol, I90_pol, I135_pol = forward_project(
            theta=theta_test, delta=delta, S_i_full=np.array([1.0, S_i_hat[0], S_i_hat[1], 0.0])
        )
        I0_vals.append(I0_pol)
        I45_vals.append(I45_pol)
        I90_vals.append(I90_pol)
        I135_vals.append(I135_pol)

    plt.plot(delta_sigma / sigma_max, I0_vals, color=["r", "g", "b"][c], linestyle="-")
    plt.plot(delta_sigma / sigma_max, I45_vals, color=["r", "g", "b"][c], linestyle="--")
    plt.plot(delta_sigma / sigma_max, I90_vals, color=["r", "g", "b"][c], linestyle="-.")
    plt.plot(delta_sigma / sigma_max, I135_vals, color=["r", "g", "b"][c], linestyle=":")
plt.xlabel(r"$\Delta\sigma/\sigma_\mathrm{max}$ (-)")
plt.ylabel("Intensity (a.u.)")
plt.text(-0.11, 1.0, "(a)", transform=plt.gca().transAxes)
plt.savefig("seeding_validation.png", dpi=300)

print("Done plotting retardance curves, now plotting heatmaps.")

stress_levels = np.linspace(0.0, sigma_max, resolution + 1)[1:]
SnRs = np.logspace(0, 2, resolution)
error_array = np.zeros((len(stress_levels), len(SnRs)))

for i, stress in tqdm(enumerate(stress_levels), total=len(stress_levels), desc="Stress level"):
    for j, SnR in tqdm(enumerate(SnRs), total=len(SnRs), desc="SnR", leave=False):
        errors = []
        for test in range(n_tests):
            relative_error = test_seeding(
                theta_test, stress, wavelengths, C_values, nu, L, S_i_hat, sigma_max, n_max, SnR
            )
            errors.append(relative_error)
        mean_error = np.median(errors)
        error_array[i, j] = mean_error


plt.subplot(223)
dSnR = np.sqrt(SnRs[1] / SnRs[0])
SnRplot = np.logspace(np.log10(SnRs[0] / dSnR), np.log10(SnRs[-1] * dSnR), len(SnRs) + 1)
stress_plot = np.linspace(stress_levels[0] / sigma_max, stress_levels[-1] / sigma_max, len(stress_levels) + 1)
plt.pcolormesh(
    SnRplot,
    stress_plot,
    error_array,
    shading="auto",
    norm=LogNorm(vmin=1e-3, vmax=1e0),
    rasterized=True,
)
plt.colorbar(label="Median error", extend="both")
plt.xscale("log")
plt.xlabel("Signal-to-noise ratio (SnR)")
plt.ylabel(r"$\Delta\sigma/\sigma_\mathrm{max}$ (-)")  # (")
plt.text(-0.32, 1.05, "(b)", transform=plt.gca().transAxes)
plt.savefig("seeding_validation.png", dpi=300)

print("Done plotting stress level heatmap, now angle heatmap.")

# GRAPH 2 ---- ANGLE VERSUS SnR

angles = np.linspace(0.0, np.pi / 2.0, resolution)  # Fraction of sigma_max
SnRs = np.logspace(0, 3, resolution)
error_array = np.zeros((len(angles), len(SnRs)))

for i, angle in tqdm(enumerate(angles), total=len(angles), desc="Angle"):
    for j, SnR in tqdm(enumerate(SnRs), total=len(SnRs), desc="SnR", leave=False):
        errors = []
        for test in range(n_tests):
            # Random stress state
            theta = angle
            delta_sigma = sigma_max * 0.5

            relative_error = test_seeding(
                theta, delta_sigma, wavelengths, C_values, nu, L, S_i_hat, sigma_max, n_max, SnR
            )
            errors.append(relative_error)
        mean_error = np.median(errors)
        error_array[i, j] = mean_error


plt.subplot(224)
dSnR = np.sqrt(SnRs[1] / SnRs[0])
SnRplot = np.logspace(np.log10(SnRs[0] / dSnR), np.log10(SnRs[-1] * dSnR), len(SnRs) + 1)
angle_plot = np.linspace(angles[0], angles[-1], len(angles) + 1)
plt.pcolormesh(
    SnRplot,
    angle_plot,
    error_array,
    shading="auto",
    norm=LogNorm(vmin=1e-3, vmax=1e0),
    rasterized=True,
)
plt.colorbar(label="Median error", extend="both")
plt.xscale("log")
plt.yticks([0, np.pi / 4, np.pi / 2], ["0", r"$\pi/4$", r"$\pi/2$"])
plt.xlabel("Signal-to-noise ratio (SnR)")
plt.ylabel(r"Angle ($\theta$)")
plt.text(-0.32, 1.05, "(c)", transform=plt.gca().transAxes)

print("Saving figure to seeding_validation.png")
plt.subplots_adjust(left=0.10, bottom=0.11, right=0.94, top=0.97, hspace=0.5, wspace=0.475)
# plt.show()
plt.savefig("seeding_validation.png", dpi=300)
plt.savefig("seeding_validation.pdf", dpi=300)
