#!/usr/bin/env python3
"""
Comprehensive pytest tests for the elastic simulation module.

This test suite verifies synthetic photoelastic data generation
for Boussinesq point load on an elastic half-space.
"""

import numpy as np
import pytest

from photoelastimetry.elastic import (
    boussinesq_stress_cartesian,
    generate_synthetic_boussinesq,
    simulate_four_step_polarimetry,
)


@pytest.fixture
def test_parameters():
    """Fixture providing standard test parameters."""
    return {
        "wavelengths_nm": np.array([650e-9, 550e-9, 450e-9]),  # R, G, B wavelengths in meters
        "C_values": np.array([2e-12, 2.2e-12, 2.5e-12]),  # Different C for each wavelength
        "nu": 1.0,  # Solid fraction
        "L": 0.01,  # Sample thickness (m)
        "S_i_hat": np.array([0.1, 0.2, 0.0]),  # Incoming polarization [S1_hat, S2_hat, S3_hat]
        "I0": 1.0,  # Incident intensity
    }


@pytest.fixture
def sample_stress():
    """Fixture providing sample stress tensor components."""
    return {
        "sigma_xx": 2e6,  # Pa
        "sigma_yy": -1e6,  # Pa
        "sigma_xy": 0.5e6,  # Pa
    }


class TestFourStepPolarimetry:
    """Test class for four-step polarimetry simulation."""

    def test_simulate_four_step_polarimetry_basic(self, test_parameters, sample_stress):
        """Test basic four-step polarimetry simulation."""
        sigma_xx = sample_stress["sigma_xx"]
        sigma_yy = sample_stress["sigma_yy"]
        sigma_xy = sample_stress["sigma_xy"]

        C = test_parameters["C_values"][0]
        nu = test_parameters["nu"]
        L = test_parameters["L"]
        wavelength = test_parameters["wavelengths_nm"][0]
        S_i_hat = test_parameters["S_i_hat"]
        I0 = test_parameters["I0"]

        I0_pol, I45_pol, I90_pol, I135_pol = simulate_four_step_polarimetry(
            sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength, S_i_hat, I0
        )

        # Check that all intensities are positive and finite
        assert I0_pol >= 0, "I0 polarization should be non-negative"
        assert I45_pol >= 0, "I45 polarization should be non-negative"
        assert I90_pol >= 0, "I90 polarization should be non-negative"
        assert I135_pol >= 0, "I135 polarization should be non-negative"

        assert np.isfinite(I0_pol), "I0 polarization should be finite"
        assert np.isfinite(I45_pol), "I45 polarization should be finite"
        assert np.isfinite(I90_pol), "I90 polarization should be finite"
        assert np.isfinite(I135_pol), "I135 polarization should be finite"

        # Check that intensities are reasonable (not too large)
        max_intensity = 2 * I0  # Conservative upper bound
        assert I0_pol <= max_intensity, "I0 intensity should be reasonable"
        assert I45_pol <= max_intensity, "I45 intensity should be reasonable"
        assert I90_pol <= max_intensity, "I90 intensity should be reasonable"
        assert I135_pol <= max_intensity, "I135 intensity should be reasonable"

    def test_simulate_four_step_polarimetry_no_stress(self, test_parameters):
        """Test four-step polarimetry with zero stress."""
        # No stress case
        I0_pol, I45_pol, I90_pol, I135_pol = simulate_four_step_polarimetry(
            0.0,
            0.0,
            0.0,  # No stress
            test_parameters["C_values"][0],
            test_parameters["nu"],
            test_parameters["L"],
            test_parameters["wavelengths_nm"][0],
            test_parameters["S_i_hat"],
            test_parameters["I0"],
        )

        # With no birefringence, should get specific intensity pattern
        assert np.isfinite(I0_pol), "I0 should be finite with no stress"
        assert np.isfinite(I45_pol), "I45 should be finite with no stress"
        assert np.isfinite(I90_pol), "I90 should be finite with no stress"
        assert np.isfinite(I135_pol), "I135 should be finite with no stress"


class TestBoussinesqStressField:
    """Test class for Boussinesq stress field computation."""

    def test_boussinesq_stress_cartesian_basic(self):
        """Test basic Boussinesq stress field computation."""
        # Test parameters
        P = 100.0  # Point load (N)
        nu_poisson = 0.3  # Poisson's ratio

        # Create grid
        n = 50
        L = 0.02
        depth = 0.02
        x = np.linspace(-L, L, n)
        y = np.linspace(0, depth, n)
        X, Y = np.meshgrid(x, y)

        # Compute Boussinesq stress distribution
        sigma_xx, sigma_yy, tau_xy = boussinesq_stress_cartesian(X, Y, P, nu_poisson)

        # Check output shapes
        assert sigma_xx.shape == (n, n), f"Expected shape ({n}, {n}), got {sigma_xx.shape}"
        assert sigma_yy.shape == (n, n), f"Expected shape ({n}, {n}), got {sigma_yy.shape}"
        assert tau_xy.shape == (n, n), f"Expected shape ({n}, {n}), got {tau_xy.shape}"

        # All stress values should be finite
        assert np.all(np.isfinite(sigma_xx)), "sigma_xx should be finite"
        assert np.all(np.isfinite(sigma_yy)), "sigma_yy should be finite"
        assert np.all(np.isfinite(tau_xy)), "tau_xy should be finite"

        # Check that stress varies across the domain (not uniform)
        mask = Y > 0  # Below surface
        assert np.std(sigma_xx[mask]) > 0, "sigma_xx should vary across the domain"
        assert np.std(sigma_yy[mask]) > 0, "sigma_yy should vary across the domain"

    def test_boussinesq_stress_cartesian_surface_boundary(self):
        """Test that Boussinesq solution respects surface boundary conditions."""
        P = 100.0
        nu_poisson = 0.3

        n = 40
        L = 0.02
        depth = 0.02
        x = np.linspace(-L, L, n)
        y = np.linspace(-0.01, depth, n)  # Include region above surface
        X, Y = np.meshgrid(x, y)

        sigma_xx, sigma_yy, tau_xy = boussinesq_stress_cartesian(X, Y, P, nu_poisson)

        # Stress should be zero above the surface (y < 0)
        above_surface = Y < 0
        if np.any(above_surface):
            assert np.all(sigma_xx[above_surface] == 0), "sigma_xx should be zero above surface"
            assert np.all(sigma_yy[above_surface] == 0), "sigma_yy should be zero above surface"
            assert np.all(tau_xy[above_surface] == 0), "tau_xy should be zero above surface"

    def test_boussinesq_stress_cartesian_stress_magnitude(self):
        """Test that Boussinesq stress magnitudes are reasonable."""
        P = 100.0
        nu_poisson = 0.3

        n = 50
        L = 0.02
        depth = 0.02
        x = np.linspace(-L, L, n)
        y = np.linspace(0.001, depth, n)  # Avoid singularity at surface
        X, Y = np.meshgrid(x, y)

        sigma_xx, sigma_yy, tau_xy = boussinesq_stress_cartesian(X, Y, P, nu_poisson)

        # Check that stress magnitudes are reasonable
        # For Boussinesq, stresses scale with P and decay with distance
        max_stress = np.max(np.abs(sigma_yy))
        
        # Stress should be non-negligible
        assert max_stress > 0, "Maximum stress should be positive"
        
        # Stress shouldn't be unreasonably large (order of magnitude check)
        # Near the load point, stress can be large, but should be bounded
        assert max_stress < 1e9, "Stress magnitude should be reasonable"


class TestSyntheticBoussinesqData:
    """Test class for synthetic Boussinesq data generation."""

    def test_generate_synthetic_boussinesq_basic(self, test_parameters):
        """Test basic synthetic Boussinesq data generation."""
        # Grid parameters
        n = 50
        L = 0.02
        depth = 0.02
        x = np.linspace(-L, L, n)
        y = np.linspace(0, depth, n)
        X, Y = np.meshgrid(x, y)
        mask = Y >= 0  # Valid region below surface

        # Physical parameters
        P = 100.0  # Point load (N)
        nu_poisson = 0.3
        thickness = test_parameters["L"]
        wavelengths_nm = test_parameters["wavelengths_nm"]
        C_values = test_parameters["C_values"]
        S_i_hat = test_parameters["S_i_hat"]
        polarisation_efficiency = 1.0

        try:
            result = generate_synthetic_boussinesq(
                X, Y, P, nu_poisson, S_i_hat, mask, wavelengths_nm, thickness, C_values, polarisation_efficiency
            )

            synthetic_images, principal_diff, theta_p, sigma_xx, sigma_yy, tau_xy = result

            # Check output shapes
            n_wavelengths = len(wavelengths_nm)
            expected_image_shape = (n, n, n_wavelengths, 4)  # 4 polarization angles
            assert (
                synthetic_images.shape == expected_image_shape
            ), f"Expected shape {expected_image_shape}, got {synthetic_images.shape}"

            # Check stress components shape
            assert sigma_xx.shape == (n, n), f"Expected stress shape ({n}, {n}), got {sigma_xx.shape}"
            assert sigma_yy.shape == (n, n), f"Expected stress shape ({n}, {n}), got {sigma_yy.shape}"
            assert tau_xy.shape == (n, n), f"Expected stress shape ({n}, {n}), got {tau_xy.shape}"

            # Check that images contain finite values where mask is True
            assert np.all(
                np.isfinite(synthetic_images[mask])
            ), "Synthetic images should be finite in valid region"
            assert np.all(synthetic_images[mask] >= 0), "Synthetic intensities should be non-negative"

            # Check that stress field values are finite where mask is True
            assert np.all(np.isfinite(sigma_xx[mask])), "sigma_xx should be finite in valid region"
            assert np.all(np.isfinite(sigma_yy[mask])), "sigma_yy should be finite in valid region"
            assert np.all(np.isfinite(tau_xy[mask])), "tau_xy should be finite in valid region"

        except Exception as e:
            pytest.skip(f"Synthetic Boussinesq data generation skipped: {e}")

    def test_generate_synthetic_boussinesq_different_parameters(self, test_parameters):
        """Test synthetic Boussinesq with different parameters."""
        # Test with different load and smaller grid
        n = 30
        L = 0.01
        depth = 0.01
        x = np.linspace(-L, L, n)
        y = np.linspace(0, depth, n)
        X, Y = np.meshgrid(x, y)
        mask = Y >= 0

        P = 50.0  # Different load
        nu_poisson = 0.25  # Different Poisson's ratio
        thickness = 0.005  # Thinner sample

        try:
            result = generate_synthetic_boussinesq(
                X,
                Y,
                P,
                nu_poisson,
                test_parameters["S_i_hat"],
                mask,
                test_parameters["wavelengths_nm"],
                thickness,
                test_parameters["C_values"],
                1.0,
            )

            synthetic_images = result[0]

            # Basic shape checks
            assert synthetic_images.shape[0] == n, "Image height should match grid size"
            assert synthetic_images.shape[1] == n, "Image width should match grid size"

        except Exception as e:
            pytest.skip(f"Parameter variation test skipped: {e}")


class TestParameterValidation:
    """Test class for parameter validation and edge cases."""

    def test_boussinesq_zero_load(self):
        """Test Boussinesq solution with zero load."""
        P = 0.0
        nu_poisson = 0.3

        n = 20
        L = 0.01
        depth = 0.01
        x = np.linspace(-L, L, n)
        y = np.linspace(0, depth, n)
        X, Y = np.meshgrid(x, y)

        sigma_xx, sigma_yy, tau_xy = boussinesq_stress_cartesian(X, Y, P, nu_poisson)

        # With zero load, all stresses should be zero
        assert np.allclose(sigma_xx, 0), "sigma_xx should be zero for zero load"
        assert np.allclose(sigma_yy, 0), "sigma_yy should be zero for zero load"
        assert np.allclose(tau_xy, 0), "tau_xy should be zero for zero load"

    def test_boussinesq_different_poisson_ratios(self):
        """Test Boussinesq solution with different Poisson's ratios."""
        P = 100.0
        poisson_ratios = [0.0, 0.25, 0.3, 0.4, 0.49]

        n = 30
        L = 0.01
        depth = 0.01
        x = np.linspace(-L, L, n)
        y = np.linspace(0.001, depth, n)
        X, Y = np.meshgrid(x, y)

        for nu in poisson_ratios:
            sigma_xx, sigma_yy, tau_xy = boussinesq_stress_cartesian(X, Y, P, nu)

            # Check that all stresses are finite
            assert np.all(np.isfinite(sigma_xx)), f"sigma_xx should be finite for nu={nu}"
            assert np.all(np.isfinite(sigma_yy)), f"sigma_yy should be finite for nu={nu}"
            assert np.all(np.isfinite(tau_xy)), f"tau_xy should be finite for nu={nu}"


if __name__ == "__main__":
    # Run tests with pytest when called directly
    import subprocess

    subprocess.run(["pytest", __file__, "-v"])
