#!/usr/bin/env python3
"""
Pytest tests for the inclined plane stress field simulation.

This test suite verifies synthetic photoelastic data generation
for an inclined plane with gravity at an angle.
"""

import numpy as np
import pytest

from photoelastimetry.generate.inclined_plane import (
    generate_synthetic_inclined_plane,
    inclined_stress_cartesian,
)


@pytest.fixture
def test_parameters():
    """Fixture providing standard test parameters."""
    return {
        "wavelengths_nm": np.array([650e-9, 550e-9, 450e-9]),  # R, G, B wavelengths in meters
        "C_values": np.array([1.5e-7, 1.5e-7, 1.5e-7]),  # Stress-optic coefficients
        "thickness": 0.01,  # Sample thickness (m)
        "S_i_hat": np.array([0.1, 0.2]),  # Incoming polarization [S1_hat, S2_hat]
        "rho": 2500,  # Density (kg/m^3)
        "g": 9.81,  # Gravity (m/s^2)
        "K0": 0.5,  # Lateral earth pressure coefficient
    }


class TestInclinedStressField:
    """Test class for inclined plane stress field computation."""

    def test_inclined_stress_cartesian_basic(self, test_parameters):
        """Test basic inclined stress field computation."""
        theta_deg = 30.0  # 30 degree inclination

        # Create grid
        n = 50
        width = 0.1
        height = 0.1
        x = np.linspace(0, width, n)
        y = np.linspace(0, height, n)
        X, Y = np.meshgrid(x, y)

        # Compute inclined stress distribution
        sigma_xx, sigma_yy, tau_xy = inclined_stress_cartesian(
            X, Y, test_parameters["rho"], test_parameters["g"], theta_deg, test_parameters["K0"]
        )

        # Check output shapes
        assert sigma_xx.shape == (n, n), f"Expected shape ({n}, {n}), got {sigma_xx.shape}"
        assert sigma_yy.shape == (n, n), f"Expected shape ({n}, {n}), got {sigma_yy.shape}"
        assert tau_xy.shape == (n, n), f"Expected shape ({n}, {n}), got {tau_xy.shape}"

        # All stress values should be finite
        assert np.all(np.isfinite(sigma_xx)), "sigma_xx should be finite"
        assert np.all(np.isfinite(sigma_yy)), "sigma_yy should be finite"
        assert np.all(np.isfinite(tau_xy)), "tau_xy should be finite"

        # Check that stress varies with depth
        assert np.std(sigma_yy) > 0, "sigma_yy should vary across the domain"

        # For inclined case, shear stress should be non-zero
        assert np.max(np.abs(tau_xy)) > 0, "tau_xy should be non-zero for inclined gravity"

    def test_inclined_stress_vertical_case(self, test_parameters):
        """Test that inclined stress reduces to lithostatic for vertical gravity."""
        theta_deg = 0.0  # Vertical (no inclination)

        n = 30
        width = 0.1
        height = 0.1
        x = np.linspace(0, width, n)
        y = np.linspace(0, height, n)
        X, Y = np.meshgrid(x, y)

        sigma_xx, sigma_yy, tau_xy = inclined_stress_cartesian(
            X, Y, test_parameters["rho"], test_parameters["g"], theta_deg, test_parameters["K0"]
        )

        # For vertical case (theta=0), shear stress should be zero
        assert np.allclose(tau_xy, 0), "tau_xy should be zero for vertical gravity"

        # Vertical stress should increase linearly with depth
        expected_sigma_yy = test_parameters["rho"] * test_parameters["g"] * Y
        assert np.allclose(sigma_yy, expected_sigma_yy, rtol=1e-10), "sigma_yy should match lithostatic for theta=0"

        # Horizontal stress should be K0 times vertical stress
        expected_sigma_xx = test_parameters["K0"] * expected_sigma_yy
        assert np.allclose(sigma_xx, expected_sigma_xx, rtol=1e-10), "sigma_xx should be K0 * sigma_yy"

    def test_inclined_stress_stress_magnitude(self, test_parameters):
        """Test that inclined stress magnitudes are reasonable."""
        theta_deg = 45.0  # 45 degree inclination

        n = 50
        width = 0.1
        height = 0.1
        x = np.linspace(0, width, n)
        y = np.linspace(0, height, n)
        X, Y = np.meshgrid(x, y)

        sigma_xx, sigma_yy, tau_xy = inclined_stress_cartesian(
            X, Y, test_parameters["rho"], test_parameters["g"], theta_deg, test_parameters["K0"]
        )

        # Check that stress magnitudes are reasonable
        max_stress = np.max(np.abs(sigma_yy))
        max_shear = np.max(np.abs(tau_xy))

        # Stress should be non-negligible
        assert max_stress > 0, "Maximum stress should be positive"
        assert max_shear > 0, "Maximum shear stress should be positive for inclined case"

        # Stress shouldn't be unreasonably large
        # For rho=2500 kg/m^3, g=9.81 m/s^2, depth=0.1 m:
        # max stress ~ rho*g*h = 2500*9.81*0.1 ~ 2.5 kPa
        expected_order = test_parameters["rho"] * test_parameters["g"] * height
        assert max_stress < 10 * expected_order, "Stress magnitude should be reasonable"
        assert max_shear < 10 * expected_order, "Shear stress magnitude should be reasonable"

    def test_inclined_stress_non_negative(self, test_parameters):
        """Test that normal stresses are non-negative (compression positive)."""
        theta_deg = 30.0

        n = 40
        width = 0.1
        height = 0.1
        x = np.linspace(0, width, n)
        y = np.linspace(0, height, n)
        X, Y = np.meshgrid(x, y)

        sigma_xx, sigma_yy, tau_xy = inclined_stress_cartesian(
            X, Y, test_parameters["rho"], test_parameters["g"], theta_deg, test_parameters["K0"]
        )

        # Normal stresses should be non-negative (no tension)
        assert np.all(sigma_xx >= 0), "sigma_xx should be non-negative"
        assert np.all(sigma_yy >= 0), "sigma_yy should be non-negative"


class TestSyntheticInclinedPlaneData:
    """Test class for synthetic inclined plane data generation."""

    def test_generate_synthetic_inclined_plane_basic(self, test_parameters):
        """Test basic synthetic inclined plane data generation."""
        # Grid parameters
        n = 50
        width = 0.1
        height = 0.1
        x = np.linspace(0, width, n)
        y = np.linspace(0, height, n)
        X, Y = np.meshgrid(x, y)

        # Physical parameters
        theta_deg = 30.0

        try:
            result = generate_synthetic_inclined_plane(
                X,
                Y,
                test_parameters["rho"],
                test_parameters["g"],
                theta_deg,
                test_parameters["K0"],
                test_parameters["S_i_hat"],
                test_parameters["wavelengths_nm"],
                test_parameters["thickness"],
                test_parameters["C_values"],
            )

            synthetic_images, principal_diff, theta_p, sigma_xx, sigma_yy, tau_xy = result

            # Check output shapes
            n_wavelengths = len(test_parameters["wavelengths_nm"])
            expected_image_shape = (n, n, n_wavelengths, 4)  # 4 polarization angles
            assert (
                synthetic_images.shape == expected_image_shape
            ), f"Expected shape {expected_image_shape}, got {synthetic_images.shape}"

            # Check stress components shape
            assert sigma_xx.shape == (n, n), f"Expected stress shape ({n}, {n}), got {sigma_xx.shape}"
            assert sigma_yy.shape == (n, n), f"Expected stress shape ({n}, {n}), got {sigma_yy.shape}"
            assert tau_xy.shape == (n, n), f"Expected stress shape ({n}, {n}), got {tau_xy.shape}"

            # Check that images contain finite values
            assert np.all(np.isfinite(synthetic_images)), "Synthetic images should be finite"
            assert np.all(synthetic_images >= 0), "Synthetic intensities should be non-negative"

            # Check that stress field values are finite
            assert np.all(np.isfinite(sigma_xx)), "sigma_xx should be finite"
            assert np.all(np.isfinite(sigma_yy)), "sigma_yy should be finite"
            assert np.all(np.isfinite(tau_xy)), "tau_xy should be finite"

            # Check that principal stress quantities are finite
            assert np.all(np.isfinite(principal_diff)), "principal_diff should be finite"
            assert np.all(np.isfinite(theta_p)), "theta_p should be finite"

        except Exception as e:
            pytest.fail(f"Synthetic inclined plane data generation failed: {e}")

    def test_generate_synthetic_inclined_plane_different_angles(self, test_parameters):
        """Test synthetic inclined plane with different inclination angles."""
        n = 30
        width = 0.05
        height = 0.05
        x = np.linspace(0, width, n)
        y = np.linspace(0, height, n)
        X, Y = np.meshgrid(x, y)

        # Test with different inclination angles
        angles = [0.0, 15.0, 30.0, 45.0, 60.0]

        for theta_deg in angles:
            try:
                result = generate_synthetic_inclined_plane(
                    X,
                    Y,
                    test_parameters["rho"],
                    test_parameters["g"],
                    theta_deg,
                    test_parameters["K0"],
                    test_parameters["S_i_hat"],
                    test_parameters["wavelengths_nm"],
                    test_parameters["thickness"],
                    test_parameters["C_values"],
                )

                synthetic_images = result[0]
                tau_xy = result[5]

                # Basic shape checks
                assert synthetic_images.shape[0] == n, "Image height should match grid size"
                assert synthetic_images.shape[1] == n, "Image width should match grid size"

                # For non-zero angles, shear stress should be non-zero
                if theta_deg > 0:
                    assert np.max(np.abs(tau_xy)) > 0, f"tau_xy should be non-zero for theta={theta_deg}°"

            except Exception as e:
                pytest.fail(f"Test failed for angle {theta_deg}°: {e}")


class TestParameterValidation:
    """Test class for parameter validation and edge cases."""

    def test_inclined_stress_zero_density(self, test_parameters):
        """Test inclined stress with zero density."""
        n = 20
        width = 0.05
        height = 0.05
        x = np.linspace(0, width, n)
        y = np.linspace(0, height, n)
        X, Y = np.meshgrid(x, y)

        sigma_xx, sigma_yy, tau_xy = inclined_stress_cartesian(X, Y, 0.0, test_parameters["g"], 30.0, 0.5)

        # With zero density, all stresses should be zero
        assert np.allclose(sigma_xx, 0), "sigma_xx should be zero for zero density"
        assert np.allclose(sigma_yy, 0), "sigma_yy should be zero for zero density"
        assert np.allclose(tau_xy, 0), "tau_xy should be zero for zero density"

    def test_inclined_stress_different_K0(self, test_parameters):
        """Test inclined stress with different K0 values."""
        n = 30
        width = 0.05
        height = 0.05
        x = np.linspace(0, width, n)
        y = np.linspace(0, height, n)
        X, Y = np.meshgrid(x, y)

        K0_values = [0.3, 0.5, 0.7, 1.0]
        theta_deg = 20.0

        for K0 in K0_values:
            sigma_xx, sigma_yy, tau_xy = inclined_stress_cartesian(
                X, Y, test_parameters["rho"], test_parameters["g"], theta_deg, K0
            )

            # Check that all stresses are finite
            assert np.all(np.isfinite(sigma_xx)), f"sigma_xx should be finite for K0={K0}"
            assert np.all(np.isfinite(sigma_yy)), f"sigma_yy should be finite for K0={K0}"
            assert np.all(np.isfinite(tau_xy)), f"tau_xy should be finite for K0={K0}"

            # Verify relationship between sigma_xx and sigma_yy
            # For vertical component: sigma_xx = K0 * sigma_yy
            # (This is approximate for inclined case, but should hold reasonably)
            g_y = test_parameters["g"] * np.cos(np.deg2rad(theta_deg))
            expected_sigma_yy = test_parameters["rho"] * g_y * Y
            expected_sigma_xx = K0 * expected_sigma_yy
            assert np.allclose(sigma_xx, expected_sigma_xx, rtol=1e-10), f"sigma_xx relationship should hold for K0={K0}"


if __name__ == "__main__":
    # Run tests with pytest when called directly
    import subprocess

    subprocess.run(["pytest", __file__, "-v"])
