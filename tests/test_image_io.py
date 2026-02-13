#!/usr/bin/env python3
"""
Comprehensive pytest tests for image processing and I/O modules.

This test suite verifies image loading, processing, and analysis functions
used in photoelastic stress analysis.
"""

import os
import tempfile

import numpy as np
import pytest

from photoelastimetry.io import load_image

# Test imports - some may not be available depending on implementation
try:
    from photoelastimetry.image import compute_principal_angle, compute_retardance, mueller_matrix

    IMAGE_MODULE_AVAILABLE = True
except ImportError:
    IMAGE_MODULE_AVAILABLE = False


@pytest.fixture
def test_parameters():
    """Fixture providing standard test parameters."""
    return {
        "wavelengths": np.array([650e-9, 550e-9, 450e-9]),  # R, G, B in meters
        "C_values": np.array([2e-12, 2.2e-12, 2.5e-12]),  # Different C for each wavelength
        "nu": 1.0,  # Solid fraction
        "L": 0.01,  # Sample thickness (m)
    }


@pytest.fixture
def sample_stress():
    """Fixture providing sample stress tensor components."""
    return {
        "sigma_xx": 2e6,  # Pa
        "sigma_yy": -1e6,  # Pa
        "sigma_xy": 0.5e6,  # Pa
    }


@pytest.fixture
def temp_directory():
    """Fixture providing a temporary directory for I/O tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.mark.skipif(not IMAGE_MODULE_AVAILABLE, reason="Image module not available")
class TestImageProcessing:
    """Test class for image processing functions."""

    def test_compute_retardance(self, test_parameters, sample_stress):
        """Test retardance computation from stress tensor."""
        sigma_xx = sample_stress["sigma_xx"]
        sigma_yy = sample_stress["sigma_yy"]
        sigma_xy = sample_stress["sigma_xy"]

        C = test_parameters["C_values"][1]
        nu = test_parameters["nu"]
        L = test_parameters["L"]
        wavelength = test_parameters["wavelengths"][1]

        delta = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength)

        # Verify formula: delta = (2*pi*C*n*L/lambda) * sqrt((sigma_xx-sigma_yy)^2 + 4*sigma_xy^2)
        psd = np.sqrt((sigma_xx - sigma_yy) ** 2 + 4 * sigma_xy**2)
        expected_delta = (2 * np.pi * C * nu * L / wavelength) * psd

        assert np.isclose(delta, expected_delta), "Retardance formula verification"
        assert delta >= 0, "Retardance should be non-negative"
        assert np.isfinite(delta), "Retardance should be finite"

    def test_compute_retardance_array_inputs(self, test_parameters):
        """Test retardance computation with array inputs."""
        # Test with arrays of stress values
        sigma_xx = np.array([1e6, 2e6, 3e6])
        sigma_yy = np.array([0.0, -1e6, -0.5e6])
        sigma_xy = np.array([0.5e6, 0.0, 1e6])

        C = test_parameters["C_values"][0]
        nu = test_parameters["nu"]
        L = test_parameters["L"]
        wavelength = test_parameters["wavelengths"][0]

        deltas = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C, nu, L, wavelength)

        # Check output shape and properties
        assert len(deltas) == len(sigma_xx), "Output should match input array length"
        assert np.all(deltas >= 0), "All retardances should be non-negative"
        assert np.all(np.isfinite(deltas)), "All retardances should be finite"

    def test_compute_principal_angle(self, sample_stress):
        """Test principal angle computation."""
        sigma_xx = sample_stress["sigma_xx"]
        sigma_yy = sample_stress["sigma_yy"]
        sigma_xy = sample_stress["sigma_xy"]

        theta = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)
        expected_theta = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)

        assert np.isclose(theta, expected_theta), "Principal angle formula verification"
        assert np.isfinite(theta), "Principal angle should be finite"

        # Check angle is in reasonable range
        assert -np.pi / 2 <= theta <= np.pi / 2, "Principal angle should be in [-π/2, π/2]"

    def test_compute_principal_angle_special_cases(self):
        """Test principal angle computation for special cases."""
        # Pure shear case (45° expected)
        theta_shear = compute_principal_angle(1e6, 1e6, 1e6)
        assert np.isclose(theta_shear, np.pi / 4, rtol=1e-6), "Pure shear should give 45°"

        # No shear case (0° expected)
        theta_no_shear = compute_principal_angle(2e6, 1e6, 0.0)
        assert np.isclose(theta_no_shear, 0.0, atol=1e-10), "No shear should give 0°"

        # Negative shear - the angle should be negative
        theta_neg_shear = compute_principal_angle(2e6, 1e6, -1e6)
        # arctan2(-2e6, 1e6) / 2 gives a negative angle
        assert theta_neg_shear < 0, "Negative shear should give negative angle"
        assert np.abs(theta_neg_shear) > 0.1, "Angle magnitude should be significant"

    def test_compute_principal_angle_array_inputs(self):
        """Test principal angle computation with array inputs."""
        sigma_xx = np.array([1e6, 2e6, 3e6])
        sigma_yy = np.array([0.0, 1e6, 2e6])
        sigma_xy = np.array([0.0, 1e6, 0.5e6])

        thetas = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)

        # Check output shape and properties
        assert len(thetas) == len(sigma_xx), "Output should match input array length"
        assert np.all(np.isfinite(thetas)), "All angles should be finite"
        assert np.all(thetas >= -np.pi / 2), "All angles should be >= -π/2"
        assert np.all(thetas <= np.pi / 2), "All angles should be <= π/2"

    def test_mueller_matrix_basic(self):
        """Test basic Mueller matrix computation."""
        theta = np.pi / 4  # 45 degrees
        delta = np.pi / 2  # 90 degrees retardance

        M = mueller_matrix(theta, delta)

        # Check matrix dimensions
        assert M.shape == (4, 4), "Mueller matrix should be 4x4"

        # Check that M[0,0] = 1 (intensity preservation for linear polarization basis)
        assert np.isclose(M[0, 0], 1.0), "M[0,0] should be 1.0"

        # Check all elements are finite
        assert np.all(np.isfinite(M)), "All Mueller matrix elements should be finite"

    def test_mueller_matrix_identity(self):
        """Test Mueller matrix for no retardance."""
        theta = 0.0
        delta = 0.0

        M = mueller_matrix(theta, delta)
        expected = np.eye(4)

        np.testing.assert_array_almost_equal(M, expected, decimal=10)

    def test_mueller_matrix_symmetries(self):
        """Test Mueller matrix symmetry properties."""
        # Test various angles and retardances
        angles = [0, np.pi / 8, np.pi / 4, np.pi / 2]
        retardances = [0, np.pi / 4, np.pi / 2, np.pi]

        for theta in angles:
            for delta in retardances:
                M = mueller_matrix(theta, delta)

                # Basic checks
                assert M.shape == (4, 4), "Mueller matrix should be 4x4"
                assert np.all(np.isfinite(M)), "All elements should be finite"
                assert np.isclose(M[0, 0], 1.0), "M[0,0] should be 1.0"


class TestImageProcessingEdgeCases:
    """Test class for edge cases and error handling."""

    @pytest.mark.skipif(not IMAGE_MODULE_AVAILABLE, reason="Image module not available")
    def test_zero_stress_cases(self, test_parameters):
        """Test image processing functions with zero stress."""
        # Zero stress should give zero retardance
        delta_zero = compute_retardance(
            0,
            0,
            0,
            test_parameters["C_values"][0],
            test_parameters["nu"],
            test_parameters["L"],
            test_parameters["wavelengths"][0],
        )
        assert np.isclose(delta_zero, 0.0, atol=1e-15), "Zero stress should give zero retardance"

        # Zero stress with shear should still give zero retardance
        theta_zero = compute_principal_angle(0, 0, 0)
        # Principal angle is undefined for zero stress, but function should handle it gracefully
        assert np.isfinite(theta_zero), "Principal angle should be finite for zero stress"

    @pytest.mark.skipif(not IMAGE_MODULE_AVAILABLE, reason="Image module not available")
    def test_extreme_values(self, test_parameters):
        """Test functions with extreme stress values."""
        # Very large stress
        large_stress = 1e9  # 1 GPa

        try:
            delta_large = compute_retardance(
                large_stress,
                0,
                0,
                test_parameters["C_values"][0],
                test_parameters["nu"],
                test_parameters["L"],
                test_parameters["wavelengths"][0],
            )
            assert np.isfinite(delta_large), "Should handle large stress values"
            assert delta_large >= 0, "Retardance should remain non-negative"

        except (OverflowError, ValueError):
            # It's acceptable if extreme values cause overflow
            pass

    def test_file_handling_errors(self, temp_directory):
        """Test I/O error handling."""
        # Test loading non-existent file
        non_existent_path = os.path.join(temp_directory, "does_not_exist.npy")
        with pytest.raises(FileNotFoundError):
            load_image(non_existent_path)

        # Test unsupported extension handling
        unsupported_path = os.path.join(temp_directory, "unsupported.xyz")
        with open(unsupported_path, "w") as f:
            f.write("dummy")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_image(unsupported_path)


if __name__ == "__main__":
    # Run tests with pytest when called directly
    import subprocess

    subprocess.run(["pytest", __file__, "-v"])
