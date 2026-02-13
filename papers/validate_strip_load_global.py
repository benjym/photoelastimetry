#!/usr/bin/env python3
"""
Validation of Global Solver using Strip Load Stress Field.

This script generates synthetic images for a strip load stress field and then
attempts to recover the stress field using the global equilibrium solver.
"""

import matplotlib.pyplot as plt
import numpy as np

from photoelastimetry.generate.strip_load import generate_synthetic_strip_load
from photoelastimetry.optimiser.equilibrium_mean_stress import recover_mean_stress
from photoelastimetry.seeding import phase_decomposed_seeding
from photoelastimetry.visualisation import print_boundary_conditions

# Parameters
width_m = 0.1  # 10 cm width (from -0.05 to 0.05)
height_m = 0.05  # 5 cm depth
resolution = 200  # pixels/m
W = int(width_m * resolution)
H = int(height_m * resolution)

# Coordinates
# Center the x-axis around 0 so the load is in the middle of the top edge
x = np.linspace(-width_m / 2, width_m / 2, W)
y = np.linspace(0, height_m, H)
X, Y = np.meshgrid(x, y)

# Strip load parameters
p_load = 1e2  # uniform pressure (Pa)
a_strip = 0.01  # half-width (m)

# Photoelastic parameters
thickness = 0.01  # 10 mm
wavelengths = np.array([650e-9, 550e-9, 450e-9])  # R, G, B (meters)
C_values = np.array([4.0e-9, 4.0e-9, 4.0e-9]) * 10  # Stress-optic coeff (Pa^-1)
polarisation_efficiency = 1.0

# polarisation_angle_deg = 0.0  # Incoming polarisation angle
# polarisation_angle_rad = np.deg2rad(polarisation_angle_deg)
# S_i_hat = np.array([np.cos(2 * polarisation_angle_rad), np.sin(2 * polarisation_angle_rad)])
S_i_hat = np.array([0, 0, 1])  # Circularly polarised light

# Valid region mask
mask = np.ones((H, W), dtype=bool)

print(f"Generating synthetic Strip Load data ({W}x{H}) for p={p_load / 1e6} MPa, a={a_strip * 1000} mm...")
synthetic_images, true_diff, true_theta, true_sxx, true_syy, true_txy = generate_synthetic_strip_load(
    X,
    Y,
    p_load,
    a_strip,
    S_i_hat,
    mask,
    wavelengths,
    thickness,
    C_values,
    polarisation_efficiency,
)

print("Data generation complete.")
print(f"Image stack shape: {synthetic_images.shape}")
print(f"Max vertical stress: {np.nanmax(np.abs(true_syy)):.2e} Pa")
print(f"Max horizontal stress: {np.nanmax(np.abs(true_sxx)):.2e} Pa")
print(f"Max shear stress: {np.nanmax(np.abs(true_txy)):.2e} Pa")

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

# Define boundary condition mask
boundary_mask = np.zeros((H, W), dtype=bool)
boundary_mask[0, :] = True  # Top
boundary_mask[:, 0] = True  # Left
boundary_mask[:, -1] = True  # Right
boundary_mask[-1, :] = True  # Bottom

# Use TRUE stress values for boundaries
b_xx = true_sxx.copy()
b_yy = true_syy.copy()
b_xy = true_txy.copy()

# free on top surface
b_yy[:, :] = np.nan
b_xx[:, :] = np.nan
b_xy[:, :] = np.nan
# b_yy[1:-1, 0] = np.nan  # Left side (excluding corners)
# b_yy[1:-1, -1] = np.nan  # Right side (excluding corners)
# b_xx[:, :] = np.nan

boundary_values = {
    "xx": b_xx,
    "yy": b_yy,
    "xy": b_xy,
}

print("\nVisualising Boundary Conditions:")
print_boundary_conditions(boundary_mask, boundary_values)
print("\n")

V_potential = np.zeros_like(Y)

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
    boundary_weight=1e6,
    regularisation_weight=0,
    external_potential=V_potential,
    boundary_values=boundary_values,
    max_iterations=10000,
    verbose=True,
    debug=True,
)

# Extract results
print("Solver complete. Extracting results...")

# Get Airy stress results
rec_sxx_airy, rec_syy_airy, rec_txy = recovered_mean_stress[0].get_stress_fields(recovered_mean_stress[1])

# Total stress
rec_sxx = rec_sxx_airy + V_potential
rec_syy = rec_syy_airy + V_potential

# Plotting
print("Plotting results...")
fig, axes = plt.subplots(4, 5, figsize=(15, 12), layout="constrained")

# Common scaling for stress maps
max_stress = max(np.max(np.abs(true_sxx)), np.max(np.abs(true_syy)))

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
plt.imshow(
    rec_sxx,
    cmap="RdBu_r",
)
plt.title("Recovered Sigma XX")
plt.colorbar()

plt.sca(axes[2, 1])
plt.imshow(
    rec_syy,
    cmap="RdBu_r",
)
plt.title("Recovered Sigma YY")
plt.colorbar()

plt.sca(axes[2, 2])
plt.imshow(rec_txy, cmap="RdBu_r", vmin=-np.max(np.abs(rec_txy)), vmax=np.max(np.abs(rec_txy)))
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
plt.imshow(error_sxx, cmap="RdBu_r")
plt.title("Error Sigma XX")
plt.colorbar()

error_syy = rec_syy - true_syy
plt.sca(axes[3, 1])
plt.imshow(error_syy, cmap="RdBu_r")
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

plt.savefig("papers/validate_strip_load.png", dpi=300)
print("Saved plot to papers/validate_strip_load.png")
