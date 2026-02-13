#!/usr/bin/env python3
"""
Validation of Global Solver using Inclined Plane Stress Field.

This script generates synthetic images for an inclined plane stress field
(rectangular mass with inclined gravity) and then attempts to recover the
stress field using the global equilibrium solver.
"""

import matplotlib.pyplot as plt
import numpy as np

from photoelastimetry.generate.inclined_plane import generate_synthetic_inclined_plane
from photoelastimetry.optimiser.equilibrium_mean_stress import recover_mean_stress
from photoelastimetry.seeding import phase_decomposed_seeding
from photoelastimetry.visualisation import print_boundary_conditions

# Parameters
width_m = 0.1  # 10 cm width
height_m = 0.1  # 10 cm height
resolution = 200  # pixels/m
W = int(width_m * resolution)
H = int(height_m * resolution)

# Coordinates
x = np.linspace(0, width_m, W)
y = np.linspace(0, height_m, H)
X, Y = np.meshgrid(x, y)

# Material parameters
rho = 2500  # kg/m^3
g = 9.81  # m/s^2
theta_deg = 30.0  # Inclination angle from vertical (degrees)
K0 = 0.5  # Lateral earth pressure coefficient

# Photoelastic parameters
thickness = 0.01  # 10 mm
wavelengths = np.array([650e-9, 550e-9, 450e-9])  # R, G, B (meters)
C_values = np.array([1e-7, 1e-7, 1e-7])  # Stress-optic coeff (Pa^-1)
# polarisation_angle_deg = 10.0  # Incoming polarisation angle
# polarisation_angle_rad = np.deg2rad(polarisation_angle_deg)
# S_i_hat = np.array([np.cos(2 * polarisation_angle_rad), np.sin(2 * polarisation_angle_rad)])
S_i_hat = np.array([0, 0, 1])  # Circularly polarised light

print(f"Generating synthetic inclined plane data ({W}x{H}) with θ={theta_deg}°...")
synthetic_images, true_diff, true_theta, true_sxx, true_syy, true_txy = generate_synthetic_inclined_plane(
    X, Y, rho, g, theta_deg, K0, S_i_hat, wavelengths, thickness, C_values
)

print("Data generation complete.")
print(f"Image stack shape: {synthetic_images.shape}")
print(f"Max vertical stress: {np.max(true_syy):.2e} Pa")
print(f"Max horizontal stress: {np.max(true_sxx):.2e} Pa")
print(f"Max shear stress: {np.max(np.abs(true_txy)):.2e} Pa")

# Use seeding to get initial stress guess
print("Computing initial stress guess using phase-decomposed seeding...")

initial_stress = phase_decomposed_seeding(
    synthetic_images,
    wavelengths,
    C_values,
    nu=1.0,  # Solid fraction
    L=thickness,
    S_i_hat=S_i_hat,
    sigma_max=None,
    n_max=6,
    K=0.8,
)
print("Initial stress guess computed.")

# Run Global Solver
print("\nRunning Global Solver...")
knot_spacing = 5

# Define boundary conditions
boundary_mask = np.zeros((H, W), dtype=bool)
boundary_mask[0, :] = True  # Top
boundary_mask[:, 0] = True  # Left
boundary_mask[:, -1] = True  # Right
boundary_mask[-1, :] = True  # Bottom

# Use TRUE stress values for boundaries to validate internal consistency
b_xx = true_sxx.copy()
b_yy = true_syy.copy()
b_xy = true_txy.copy()

# For inclined plane, we can set boundary conditions similar to lithostatic
# but accounting for the shear stress on boundaries

# Free tangential stresses on sides
b_yy[1:-1, 0] = np.nan  # Left side (excluding corners)
b_yy[1:-1, -1] = np.nan  # Right side (excluding corners)

# Free normal stress on all boundaries (to be solved)
b_xx[:, :] = np.nan

# Shear stress on boundaries is set by the true solution (from inclined gravity)
# We keep b_xy as is (from true_txy)

boundary_values = {
    "xx": b_xx,
    "yy": b_yy,
    "xy": b_xy,
}

print("\nVisualising Boundary Conditions:")
print_boundary_conditions(boundary_mask, boundary_values)
print("\n")

# Body force potential for inclined gravity
# The potential V must satisfy: dV/dx = rho*g*sin(theta), dV/dy = rho*g*cos(theta)
# This means V cannot be a simple function of Y alone (as in lithostatic case)
# For the equilibrium equations:
# d(sigma_xx)/dx + d(tau_xy)/dy = -rho*g*sin(theta)
# d(tau_xy)/dx + d(sigma_yy)/dy = -rho*g*cos(theta)
#
# The Airy stress function approach with external potential V works when:
# sigma_xx_total = sigma_xx_airy + V
# sigma_yy_total = sigma_yy_airy + V
# tau_xy_total = tau_xy_airy
#
# For this to satisfy equilibrium with inclined gravity:
# dV/dx = -rho*g*sin(theta) => V = -rho*g*sin(theta)*x + f(y)
# dV/dy = -rho*g*cos(theta) => V = -rho*g*cos(theta)*y + g(x)
# These are compatible only if we use V = -rho*g*(sin(theta)*x + cos(theta)*y)

theta_rad = np.deg2rad(theta_deg)
V_potential = rho * g * (np.sin(theta_rad) * X + np.cos(theta_rad) * Y)

initial_diff = np.sqrt(
    (initial_stress[:, :, 0] - initial_stress[:, :, 1]) ** 2 + 4 * initial_stress[:, :, 2] ** 2
)
initial_theta = 0.5 * np.arctan2(
    2 * initial_stress[:, :, 2], initial_stress[:, :, 0] - initial_stress[:, :, 1]
)

recovered_mean_stress = recover_mean_stress(
    initial_diff,
    initial_theta,
    knot_spacing=knot_spacing,
    spline_degree=3,
    boundary_mask=boundary_mask,
    boundary_weight=1,
    regularisation_weight=0,
    regularisation_order=2,
    external_potential=V_potential,
    boundary_values=boundary_values,
    max_iterations=10000,
    verbose=True,
    debug=True,
)

# Extract results
print("Solver complete. Extracting results...")

rec_sxx_airy, rec_syy_airy, rec_txy = recovered_mean_stress[0].get_stress_fields(recovered_mean_stress[1])

# Add potential back to normal stresses to get total stress
rec_sxx = rec_sxx_airy + V_potential
rec_syy = rec_syy_airy + V_potential

# Plotting
print("Plotting results...")
fig, axes = plt.subplots(4, 5, figsize=(15, 12), layout="constrained")

# Row 1: Ground Truth
plt.sca(axes[0, 0])
plt.imshow(true_sxx, cmap="RdBu_r")
plt.title("True Sigma XX")
plt.colorbar()

plt.sca(axes[0, 1])
plt.imshow(true_syy, cmap="RdBu_r")
plt.title("True Sigma YY")
plt.colorbar()

plt.sca(axes[0, 2])
plt.imshow(true_txy, cmap="RdBu_r", vmin=-np.max(np.abs(true_txy)), vmax=np.max(np.abs(true_txy)))
plt.title("True Tau XY")
plt.colorbar()

plt.sca(axes[0, 3])
plt.imshow(true_diff, cmap="viridis")
plt.title("True Principal Diff")
plt.colorbar()

plt.sca(axes[0, 4])
plt.imshow(true_theta, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
plt.title("True Principal Angle")
plt.colorbar()

# Row 2: Initial Guess
plt.sca(axes[1, 0])
plt.imshow(initial_stress[:, :, 0], cmap="RdBu_r")
plt.title("Initial Guess Sigma XX")
plt.colorbar()

plt.sca(axes[1, 1])
plt.imshow(initial_stress[:, :, 1], cmap="RdBu_r")
plt.title("Initial Guess Sigma YY")
plt.colorbar()

plt.sca(axes[1, 2])
plt.imshow(initial_stress[:, :, 2], cmap="RdBu_r")
plt.title("Initial Guess Tau XY")
plt.colorbar()

plt.sca(axes[1, 3])
plt.imshow(initial_diff, cmap="viridis")
plt.title("Initial Guess Principal Diff")
plt.colorbar()

plt.sca(axes[1, 4])
plt.imshow(initial_theta, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
plt.title("Initial Guess Principal Angle")
plt.colorbar()

# Row 3: Recovered
plt.sca(axes[2, 0])
plt.imshow(rec_sxx, cmap="RdBu_r")
plt.title("Recovered Sigma XX")
plt.colorbar()

plt.sca(axes[2, 1])
plt.imshow(rec_syy, cmap="RdBu_r")
plt.title("Recovered Sigma YY")
plt.colorbar()

plt.sca(axes[2, 2])
plt.imshow(rec_txy, cmap="RdBu_r", vmin=-np.max(np.abs(true_txy)), vmax=np.max(np.abs(true_txy)))
plt.title("Recovered Tau XY")
plt.colorbar()

plt.sca(axes[2, 3])
recovered_diff = np.sqrt((rec_sxx - rec_syy) ** 2 + 4 * rec_txy**2)
plt.imshow(recovered_diff, cmap="viridis")
plt.title("Recovered Principal Diff")
plt.colorbar()

plt.sca(axes[2, 4])
recovered_theta = 0.5 * np.arctan2(2 * rec_txy, rec_sxx - rec_syy)
plt.imshow(recovered_theta, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
plt.title("Recovered Principal Angle")
plt.colorbar()

# Row 4: Error Maps
error_sxx = rec_sxx - true_sxx
plt.sca(axes[3, 0])
plt.imshow(error_sxx, cmap="RdBu_r", vmin=-np.max(np.abs(error_sxx)), vmax=np.max(np.abs(error_sxx)))
plt.title("Error Sigma XX")
plt.colorbar()

error_syy = rec_syy - true_syy
plt.sca(axes[3, 1])
plt.imshow(error_syy, cmap="RdBu_r", vmin=-np.max(np.abs(error_syy)), vmax=np.max(np.abs(error_syy)))
plt.title("Error Sigma YY")
plt.colorbar()

error_tau = rec_txy - true_txy
plt.sca(axes[3, 2])
plt.imshow(error_tau, cmap="RdBu_r", vmin=-np.max(np.abs(error_tau)), vmax=np.max(np.abs(error_tau)))
plt.title("Error Tau XY")
plt.colorbar()

error_diff = recovered_diff - true_diff
plt.sca(axes[3, 3])
plt.imshow(error_diff, cmap="viridis", vmin=-np.max(np.abs(error_diff)), vmax=np.max(np.abs(error_diff)))
plt.title("Error Principal Diff")
plt.colorbar()

plt.sca(axes[3, 4])
plt.imshow(recovered_theta - true_theta, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
plt.title("Error Principal Angle")
plt.colorbar()

plt.savefig("papers/validate_inclined_plane.png", dpi=300)
print("Saved plot to papers/validate_inclined_plane.png")
