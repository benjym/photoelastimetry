#!/usr/bin/env python3
"""
Validation of Global Solver using Strip Load Stress Field.

This script generates synthetic images for a strip load stress field and then
attempts to recover the stress field using the global equilibrium solver.
"""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from plot_config import configure_plots

from photoelastimetry.generate.strip_load import generate_synthetic_strip_load
from photoelastimetry.optimise import recover_mean_stress
from photoelastimetry.seeding import phase_decomposed_seeding
from photoelastimetry.visualisation import print_boundary_conditions

configure_plots()

cb_fraction = 0.2
cb_pad = 0.04
cb_shrink = 1.0

# Parameters
width_m = 0.1  # 10 cm width (from -0.05 to 0.05)
height_m = 0.05  # 5 cm depth
resolution = 2000  # pixels/m
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
knot_spacing = 1

# Define boundary condition mask
boundary_mask = np.zeros((H, W), dtype=bool)
boundary_mask[0, :] = True  # Top
boundary_mask[:, 0] = True  # Left
boundary_mask[:, -1] = True  # Right
boundary_mask[-1, :] = True  # Bottom

# Use true stresses on the boundary only; keep interior unconstrained.
b_xx = np.full((H, W), np.nan)
b_yy = np.full((H, W), np.nan)
b_xy = np.full((H, W), np.nan)

b_xx[boundary_mask] = true_sxx[boundary_mask]
b_yy[boundary_mask] = true_syy[boundary_mask]
b_xy[boundary_mask] = true_txy[boundary_mask]

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
    boundary_weight=1e1,
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
fig, axes = plt.subplots(5, 3, figsize=(6.04, 4.6), constrained_layout=True)

# Common scaling for stress maps
max_stress = max(np.max(np.abs(true_sxx)), np.max(np.abs(true_syy)))

recovered_diff = np.sqrt((rec_sxx - rec_syy) ** 2 + 4 * rec_txy**2)
recovered_theta = 0.5 * np.arctan2(2 * rec_txy, rec_sxx - rec_syy)
error_sxx = rec_sxx - true_sxx
error_syy = rec_syy - true_syy
error_tau = rec_txy - true_txy
error_diff = recovered_diff - true_diff
error_theta = recovered_theta - true_theta

print(
    "Mean errors [Pa]: "
    f"xx={np.nanmean(error_sxx):.3e}, yy={np.nanmean(error_syy):.3e}, "
    f"tau={np.nanmean(error_tau):.3e}, dSigma={np.nanmean(error_diff):.3e}"
)
print(
    "Std errors  [Pa]: "
    f"xx={np.nanstd(error_sxx):.3e}, yy={np.nanstd(error_syy):.3e}, "
    f"tau={np.nanstd(error_tau):.3e}, dSigma={np.nanstd(error_diff):.3e}"
)
print(
    "Boundary MAE [Pa]: "
    f"xx={np.nanmean(np.abs(error_sxx[boundary_mask])):.3e}, "
    f"yy={np.nanmean(np.abs(error_syy[boundary_mask])):.3e}"
)

# Columns: true / recovered / error
# Row 1: sigma_xx
im0 = axes[0, 0].imshow(true_sxx, cmap="inferno", vmin=0, vmax=max_stress)
cb = fig.colorbar(im0, ax=axes[0, 0], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
cb.set_label(r"$\sigma_{xx}$ (Pa)")

im1 = axes[0, 1].imshow(rec_sxx, cmap="inferno", vmin=0, vmax=max_stress)
cb = fig.colorbar(im1, ax=axes[0, 1], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
cb.set_label(r"$\sigma_{xx}$ (Pa)")

im2 = axes[0, 2].imshow(error_sxx, cmap="inferno")
cb = fig.colorbar(im2, ax=axes[0, 2], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
cb.set_label("$\\sigma_{xx}$ error\n(Pa)")
cb.ax.yaxis.label.set_multialignment("center")

# Row 2: sigma_yy
im3 = axes[1, 0].imshow(true_syy, cmap="inferno", vmin=0, vmax=max_stress)
cb = fig.colorbar(im3, ax=axes[1, 0], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
cb.set_label(r"$\sigma_{yy}$ (Pa)")

im4 = axes[1, 1].imshow(rec_syy, cmap="inferno", vmin=0, vmax=max_stress)
cb = fig.colorbar(im4, ax=axes[1, 1], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
cb.set_label(r"$\sigma_{yy}$ (Pa)")

im5 = axes[1, 2].imshow(error_syy, cmap="inferno")
cb = fig.colorbar(im5, ax=axes[1, 2], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
cb.set_label("$\\sigma_{yy}$ error\n(Pa)")
cb.ax.yaxis.label.set_multialignment("center")

# Row 3: tau_xy
im6 = axes[2, 0].imshow(true_txy, cmap="inferno")
cb = fig.colorbar(im6, ax=axes[2, 0], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
cb.set_label(r"$\tau_{xy}$ (Pa)")

im7 = axes[2, 1].imshow(rec_txy, cmap="inferno")
cb = fig.colorbar(im7, ax=axes[2, 1], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
cb.set_label(r"$\tau_{xy}$ (Pa)")

tau_err_scale = 1e-14
im8 = axes[2, 2].imshow(error_tau / tau_err_scale, cmap="inferno")
cb = fig.colorbar(im8, ax=axes[2, 2], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
cb.set_label("$\\tau_{xy}$ error\n($10^{-14}$ Pa)")
cb.ax.yaxis.label.set_multialignment("center")

# Row 4: principal difference
im9 = axes[3, 0].imshow(true_diff, cmap="inferno")
cb = fig.colorbar(im9, ax=axes[3, 0], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
cb.set_label(r"$\Delta\sigma$ (Pa)")

im10 = axes[3, 1].imshow(recovered_diff, cmap="inferno")
cb = fig.colorbar(im10, ax=axes[3, 1], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
cb.set_label(r"$\Delta\sigma$ (Pa)")

diff_err_scale = 1e-14
im11 = axes[3, 2].imshow(error_diff / diff_err_scale, cmap="inferno")
cb = fig.colorbar(im11, ax=axes[3, 2], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink)
cb.set_label("$\\Delta\\sigma$ error\n($10^{-14}$ Pa)")
cb.ax.yaxis.label.set_multialignment("center")

# Row 5: principal angle
im12 = axes[4, 0].imshow(true_theta, cmap="twilight", vmin=-np.pi / 2, vmax=np.pi / 2)
cb = fig.colorbar(
    im12, ax=axes[4, 0], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink, ticks=[-np.pi / 2, 0, np.pi / 2]
)
cb.set_label(r"$\theta$ (rad)")
cb.ax.set_yticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])

im13 = axes[4, 1].imshow(recovered_theta, cmap="twilight", vmin=-np.pi / 2, vmax=np.pi / 2)
cb = fig.colorbar(
    im13, ax=axes[4, 1], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink, ticks=[-np.pi / 2, 0, np.pi / 2]
)
cb.set_label(r"$\theta$ (rad)")
cb.ax.set_yticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])

im14 = axes[4, 2].imshow(error_theta, cmap="twilight", vmin=-np.pi / 2, vmax=np.pi / 2)
cb = fig.colorbar(
    im14, ax=axes[4, 2], fraction=cb_fraction, pad=cb_pad, shrink=cb_shrink, ticks=[-np.pi / 2, 0, np.pi / 2]
)
cb.set_label("$\\theta$ err\n(rad)")
cb.ax.yaxis.label.set_multialignment("center")
cb.ax.set_yticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])

panel_labels = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
    "(e)",
    "(f)",
    "(g)",
    "(h)",
    "(i)",
    "(j)",
    "(k)",
    "(l)",
    "(m)",
    "(n)",
    "(o)",
]

for idx, ax in enumerate(axes.ravel()):
    ax.text(-0.15, 1.15, panel_labels[idx], transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])

# Keep ticks only on bottom-left panel
axes[4, 0].set_xticks([1, W])
axes[4, 0].set_yticks([1, H])
axes[4, 0].set_xlabel("Width (px)")
axes[4, 0].set_ylabel("Height (px)")

plt.savefig("papers/validate_strip_load.pdf", dpi=300)
print("Saved plot to papers/validate_strip_load.pdf")

try:
    shutil.copy(
        "papers/validate_strip_load.pdf",
        os.path.expanduser(
            "~/Dropbox/Apps/Overleaf/RGB Granular Photoelasticity/images/validate_strip_load.pdf"
        ),
    )
    print("Copied to Dropbox")
except Exception as e:
    print(f"Could not copy to dropbox: {e}")
