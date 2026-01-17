"""
Example script demonstrating the optimization debugging visualization.

This script shows how to use the new debugging features to visualize
how the optimizer converges to the stress tensor solution and how the
predicted Stokes parameters evolve during optimization.
"""

import matplotlib.pyplot as plt
import numpy as np

from photoelastimetry.optimiser.stokes import (
    compute_normalized_stokes,
    compute_stokes_components,
    predict_stokes,
    recover_stress_tensor,
    recover_stress_tensor_live,
)
from photoelastimetry.plotting import plot_optimization_history


def create_synthetic_data():
    """Create synthetic measurement data for testing."""
    # True stress state (unknown to optimizer)
    # Use larger stresses to create measurable photoelastic effect
    sigma_xx_true = 500000.0  # Pa (500 kPa)
    sigma_yy_true = 200000.0  # Pa (200 kPa)
    sigma_xy_true = 100000.0  # Pa (100 kPa)

    # Material and optical properties
    wavelengths = np.array([650e-9, 532e-9, 473e-9])  # R, G, B wavelengths (m)
    C_values = np.array([2.3e-10, 2.5e-10, 2.7e-10])  # Stress-optic coefficients (1/Pa)
    nu = 1.0  # Solid fraction
    L = 0.01  # Sample thickness (m)
    S_i_hat = np.array([1.0, 0.0])  # Incoming linearly polarized light (horizontal)

    # Generate "measured" Stokes components by forward prediction
    S_m_hat = np.zeros((3, 2))
    for c in range(3):
        S_m_hat[c] = predict_stokes(
            sigma_xx_true, sigma_yy_true, sigma_xy_true, C_values[c], nu, L, wavelengths[c], S_i_hat
        )

    # Add small noise to make it realistic
    np.random.seed(42)
    S_m_hat += np.random.normal(0, 0.001, S_m_hat.shape)

    return S_m_hat, wavelengths, C_values, nu, L, S_i_hat, (sigma_xx_true, sigma_yy_true, sigma_xy_true)


def example_1_static_plot():
    """Example 1: Generate a static plot after optimization completes."""
    print("\n" + "=" * 60)
    print("Example 1: Static optimization history plot")
    print("=" * 60)

    # Create synthetic data
    S_m_hat, wavelengths, C_values, nu, L, S_i_hat, true_stress = create_synthetic_data()

    print("\nTrue stress state:")
    print(f"  σ_xx = {true_stress[0]:.2f} Pa")
    print(f"  σ_yy = {true_stress[1]:.2f} Pa")
    print(f"  σ_xy = {true_stress[2]:.2f} Pa")

    # Run optimization with history tracking
    print("\nRunning optimization with history tracking...")
    initial_guess = np.array([100000.0, 100000.0, 10000.0])  # Deliberately poor initial guess

    stress_recovered, success, history = recover_stress_tensor(
        S_m_hat, wavelengths, C_values, nu, L, S_i_hat, initial_guess=initial_guess, track_history=True
    )

    print(f"\nOptimization {'succeeded' if success else 'failed'}")
    best_path = history["all_paths"][history["best_path_index"]]
    print(f"Number of paths explored: {len(history['all_paths'])}")
    print(f"Best path iterations: {len(best_path['residuals'])}")
    print(f"\nRecovered stress state:")
    print(f"  σ_xx = {stress_recovered[0]:.2f} Pa")
    print(f"  σ_yy = {stress_recovered[1]:.2f} Pa")
    print(f"  σ_xy = {stress_recovered[2]:.2f} Pa")
    print(f"\nFinal residual: {best_path['residuals'][-1]:.2e}")

    # Check if we found an equivalent solution (photoelastic ambiguity)
    # The photoelastic effect has ambiguity: swapping σ_xx <-> σ_yy and flipping σ_xy gives same result
    if abs(stress_recovered[0] - true_stress[1]) < abs(stress_recovered[0] - true_stress[0]) * 0.1:
        print("\nNote: Found equivalent solution due to photoelastic ambiguity")
        print(f"  (σ_xx, σ_yy, σ_xy) ≈ (σ_yy_true, σ_xx_true, -σ_xy_true)")
        print(f"  This gives identical Stokes parameters!")

    # Create detailed static plot
    print("\nGenerating optimization history plot...")
    fig = plot_optimization_history(history, S_m_hat, filename="optimization_history.png")
    print("Saved plot to: optimization_history.png")

    return history, S_m_hat


def example_2_live_plot():
    """Example 2: Watch optimization progress in real-time."""
    print("\n" + "=" * 60)
    print("Example 2: Live optimization visualization")
    print("=" * 60)
    print("\nThis will open a live-updating plot window.")
    print("Watch how the optimizer searches the stress space!")

    # Create synthetic data
    S_m_hat, wavelengths, C_values, nu, L, S_i_hat, true_stress = create_synthetic_data()

    print("\nTrue stress state:")
    print(f"  σ_xx = {true_stress[0]:.2f} Pa")
    print(f"  σ_yy = {true_stress[1]:.2f} Pa")
    print(f"  σ_xy = {true_stress[2]:.2f} Pa")

    # Run with live plotting (updates every 5 iterations)
    print("\nStarting live optimization...")
    initial_guess = np.array([100000.0, 100000.0, 10000.0])

    stress_recovered, success, history, fig = recover_stress_tensor_live(
        S_m_hat,
        wavelengths,
        C_values,
        nu,
        L,
        S_i_hat,
        initial_guess=initial_guess,
        update_interval=5,  # Update plot every 5 iterations
    )

    print(f"\nOptimization completed!")
    print(
        f"Recovered stress: σ_xx={stress_recovered[0]:.2f}, "
        f"σ_yy={stress_recovered[1]:.2f}, σ_xy={stress_recovered[2]:.2f} Pa"
    )
    print(f"Final residual: {history['residuals'][-1]:.2e}")

    # Keep window open
    print("\nClose the plot window to continue...")
    plt.show()

    return stress_recovered, history


def example_3_compare_initial_guesses():
    """Example 3: Compare optimization paths from different initial guesses."""
    print("\n" + "=" * 60)
    print("Example 3: Compare different initial guesses")
    print("=" * 60)

    # Create synthetic data
    S_m_hat, wavelengths, C_values, nu, L, S_i_hat, true_stress = create_synthetic_data()

    initial_guesses = [
        np.array([100000.0, 100000.0, 10000.0]),
        np.array([800000.0, 800000.0, 200000.0]),
        np.array([300000.0, 600000.0, -50000.0]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Comparison of Different Initial Guesses", fontsize=14, fontweight="bold")

    colors = ["blue", "red", "green"]

    for i, (guess, color) in enumerate(zip(initial_guesses, colors)):
        print(f"\nTesting initial guess {i+1}: [{guess[0]:.0f}, {guess[1]:.0f}, {guess[2]:.0f}]")

        stress, success, history = recover_stress_tensor(
            S_m_hat, wavelengths, C_values, nu, L, S_i_hat, initial_guess=guess, track_history=True
        )

        best_path = history["all_paths"][history["best_path_index"]]
        stress_params = best_path["stress_params"]
        residuals = best_path["residuals"]
        iterations = np.arange(len(residuals))

        # Plot stress component evolution
        axes[0, 0].plot(iterations, stress_params[:, 0], color=color, alpha=0.7, label=f"Guess {i+1}")
        axes[0, 0].set_ylabel("σ_xx (Pa)")
        axes[0, 0].set_title("σ_xx Evolution")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(iterations, stress_params[:, 1], color=color, alpha=0.7, label=f"Guess {i+1}")
        axes[0, 1].set_ylabel("σ_yy (Pa)")
        axes[0, 1].set_title("σ_yy Evolution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(iterations, stress_params[:, 2], color=color, alpha=0.7, label=f"Guess {i+1}")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("σ_xy (Pa)")
        axes[1, 0].set_title("σ_xy Evolution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot residual evolution
        axes[1, 1].semilogy(iterations, residuals, color=color, alpha=0.7, label=f"Guess {i+1}")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Residual (log scale)")
        axes[1, 1].set_title("Residual Evolution")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        print(f"  Final: [{stress[0]:.2f}, {stress[1]:.2f}, {stress[2]:.2f}] Pa")
        print(f"  Iterations: {len(residuals)}, Final residual: {residuals[-1]:.2e}")

    # Add true values as horizontal lines
    for ax, idx, name in [(axes[0, 0], 0, "σ_xx"), (axes[0, 1], 1, "σ_yy"), (axes[1, 0], 2, "σ_xy")]:
        ax.axhline(
            true_stress[idx], color="black", linestyle="--", linewidth=2, alpha=0.5, label="True value"
        )
        ax.legend()

    plt.tight_layout()
    plt.savefig("initial_guess_comparison.png", dpi=150)
    print("\nSaved comparison plot to: initial_guess_comparison.png")
    plt.show()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Optimization Debugging Examples")
    print("=" * 60)
    print("\nThese examples demonstrate how to visualize the stress")
    print("tensor optimization process and debug convergence issues.")

    # Example 1: Static plot after optimization
    example_1_static_plot()

    # Example 2: Live updating plot (interactive) - TODO: update for new history format
    # response = input("\nRun live plotting example? (y/n): ").strip().lower()
    # if response == "y":
    #     example_2_live_plot()

    # Example 3: Compare initial guesses
    response = input("\nCompare different initial guesses? (y/n): ").strip().lower()
    if response == "y":
        example_3_compare_initial_guesses()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
