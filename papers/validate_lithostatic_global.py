#!/usr/bin/env python3
"""
Validation of Global Solver using Lithostatic Stress Field.

This script generates synthetic images for a lithostatic stress field and then
attempts to recover the stress field using the global equilibrium solver.
"""

import matplotlib.pyplot as plt
import numpy as np

from photoelastimetry.generate.lithostatic import generate_synthetic_lithostatic

# from photoelastimetry.optimiser.equilibrium import recover_stress_global
from photoelastimetry.optimiser.equilibrium_mean_stress import recover_mean_stress
from photoelastimetry.seeding import phase_decomposed_seeding
from photoelastimetry.visualisation import print_boundary_conditions

# Parameters
width_m = 0.05  # width (m)
height_m = 0.05  # height (m)
resolution = 200  # pixels/m
W = int(width_m * resolution)
H = int(height_m * resolution)

# Coordinates
x = np.linspace(0, width_m, W)
y = np.linspace(0, height_m, H)
X, Y = np.meshgrid(x, y)

# Material parameters
# We use a very high density to simulate manageable fringes for validation
# This is equivalent to high-g or very deep sample
rho = 2500  # kg/m^3
g = 9.81  # m/s^2
K0 = 0.5  # Lateral earth pressure coefficient

# Photoelastic parameters
thickness = 0.01  # 10 mm
wavelengths = np.array([650e-9, 550e-9, 450e-9])  # R, G, B (meters)
C_values = np.array([1.5e-7, 1.5e-7, 1.5e-7])  # Stress-optic coeff (Pa^-1)
polarisation_angle_deg = 10.0  # Incoming polarisation angle
polarisation_angle_rad = np.deg2rad(polarisation_angle_deg)
# S_i_hat = np.array([np.cos(2 * polarisation_angle_rad), np.sin(2 * polarisation_angle_rad), 0])
S_i_hat = np.array([0, 0, 1])  # Incoming circularly polarised light

print(f"Generating synthetic lithostatic data ({W}x{H})...")
synthetic_images, true_diff, true_theta, true_sxx, true_syy, true_txy = generate_synthetic_lithostatic(
    X, Y, rho, g, K0, S_i_hat, wavelengths, thickness, C_values
)

print("Data generation complete.")
print(f"Image stack shape: {synthetic_images.shape}")
print(f"Max vertical stress: {np.max(true_syy):.2e} Pa")
print(f"Max horizontal stress: {np.max(true_sxx):.2e} Pa")

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
    # n_jobs=-1,
)
print("Initial stress guess computed.")

# # Run Global Solver
print("\nRunning Global Solver...")
# We use a relatively coarse knot spacing for speed in this debug script
knot_spacing = 5

# Define boundary condition: Zero stress component normal to the top surface
# For top surface (y=0, n=(0,-1)):
# sigma_yy = 0, tau_xy = 0 (free surface)
# sigma_xx is free
boundary_mask = np.zeros((H, W), dtype=bool)
boundary_mask[0, :] = True  # Top

# Optional: Add Side/Bottom boundaries with partial constraints
boundary_mask[:, 0] = True  # Left
boundary_mask[:, -1] = True  # Right
boundary_mask[-1, :] = True  # Bottom

# Use TRUE stress values for boundaries to debug convergence
b_xx = true_sxx.copy()
b_yy = true_syy.copy()
b_xy = true_txy.copy()

# 1. Tangential Stress on Sides (sigma_yy) should be Free
# But ensure we Don't un-constrain the corners where Top/Bottom normal conditions apply
b_yy[1:-1, 0] = np.nan  # Left side (excluding corners)
b_yy[1:-1, -1] = np.nan  # Right side (excluding corners)

# 2. Tangential Stress on Top/Bottom (sigma_xx) should be Free
# And Reaction Force (sigma_xx) on Sides should also be Free (solved for)
# So sigma_xx is free on ALL boundaries
b_xx[:, :] = np.nan

boundary_values = {
    "xx": b_xx,
    "yy": b_yy,
    "xy": b_xy,
}

print("\nVisualising Boundary Conditions:")
print_boundary_conditions(boundary_mask, boundary_values)
print("\n")

# Body force potential for lithostatic load
# We use Compressive (+) convention to match generation
# Equilibrium: d(Syy)/dy = rho*g => V = rho*g*y
V_potential = rho * g * Y

initial_diff = np.abs(initial_stress[:, :, 0] - initial_stress[:, :, 1])
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

# The solver returns the B-spline and coefficients
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
plt.imshow(true_txy, cmap="RdBu_r")
plt.title("True Tau XY")
plt.colorbar()

plt.sca(axes[0, 3])
plt.imshow(true_diff, cmap="viridis")
plt.title("True Principal Diff")
plt.colorbar()

plt.sca(axes[0, 4])
plt.imshow(true_theta, cmap="hsv", vmin=0, vmax=np.pi / 2.0)
plt.title("True Principal Angle")
plt.colorbar()

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
plt.imshow(initial_theta, cmap="hsv", vmin=0, vmax=np.pi / 2.0)
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
plt.imshow(rec_txy, cmap="RdBu_r")
plt.title("Recovered Tau XY")
plt.colorbar()

plt.sca(axes[2, 3])
recovered_diff = np.abs(rec_sxx - rec_syy)
plt.imshow(recovered_diff, cmap="viridis")
plt.title("Recovered Principal Diff")
plt.colorbar()

plt.sca(axes[2, 4])
recovered_theta = 0.5 * np.arctan2(2 * rec_txy, rec_sxx - rec_syy)
plt.imshow(recovered_theta, cmap="hsv", vmin=0, vmax=np.pi / 2.0)
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
plt.imshow(recovered_theta - true_theta, cmap="hsv", vmin=0, vmax=np.pi / 2.0)
plt.title("Error Principal Angle")
plt.colorbar()


# plt.tight_layout()
plt.savefig("papers/validate_lithostatic.png", dpi=300)
# print("Saved plot to validate_lithostatic_debug.png")
# plt.show()
