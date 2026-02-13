"""
Tests for the main.py module.

This module tests the main functions for converting between images and stress maps,
as well as de-mosaicing raw polarimetric images.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import json5
import numpy as np
import pytest

from photoelastimetry import io, main


class TestImageToStress:
    """Tests for image_to_stress function."""

    def test_image_to_stress_with_input_filename(self):
        """Test image_to_stress with input_filename parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic input image (H, W, colors, angles)
            data = np.random.rand(1, 2, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            # Create params dict
            params = {
                "input_filename": input_file,
                "C": 3e-9,
                "thickness": 0.01,
                "wavelengths": [650, 550, 450],
                "S_i_hat": [1.0, 0.0, 0.0],
                "debug": False,
                "solver": "intensity",
            }

            # Run image_to_stress
            stress_map = main.image_to_stress(params)

            # Check output shape (should be 3D with stress components)
            assert stress_map.ndim == 3, "Stress map should be 3D"
            assert stress_map.shape[0] == 1, "Height should match input"
            assert stress_map.shape[1] == 2, "Width should match input"
            assert stress_map.shape[2] == 3, "Should have 3 stress components"

    def test_image_to_stress_with_output_filename(self):
        """Test image_to_stress saves output when output_filename is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic input image
            data = np.random.rand(2, 2, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            output_file = os.path.join(tmpdir, "test_output.tiff")
            io.save_image(input_file, data)

            params = {
                "input_filename": input_file,
                "C": 3e-9,
                "thickness": 0.01,
                "wavelengths": [650, 550, 450],
                "S_i_hat": [1.0, 0.0],
                "debug": False,
                "output_filename": output_file,
                "solver": "intensity",
            }

            main.image_to_stress(params)

            # Check that output file was created
            assert os.path.exists(output_file), "Output file should be created"

            # Load and verify
            loaded, metadata = io.load_image(output_file)
            assert loaded.shape == (2, 2, 3), "Output should be 3D stress tensor"

    def test_image_to_stress_with_crop(self):
        """Test image_to_stress with crop parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic input image
            data = np.random.rand(20, 20, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = {
                "input_filename": input_file,
                "crop": [5, 7, 5, 7],  # [x1, x2, y1, y2]
                "C": 3e-9,
                "thickness": 0.01,
                "wavelengths": [650, 550, 450],
                "S_i_hat": [1.0, 0.0, 0.0],
                "debug": False,
                "solver": "intensity",
            }

            stress_map = main.image_to_stress(params)

            # Check cropped dimensions
            assert stress_map.shape[0] == 2, "Cropped height should be 2"
            assert stress_map.shape[1] == 2, "Cropped width should be 2"
            assert stress_map.shape[2] == 3, "Should have 3 stress components"

    def test_image_to_stress_with_binning(self):
        """Test image_to_stress with binning parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic input image
            data = np.random.rand(12, 12, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = {
                "input_filename": input_file,
                "binning": 4,
                "C": 3e-9,
                "thickness": 0.01,
                "wavelengths": [650, 550, 450],
                "S_i_hat": [1.0, 0.0, 0.0],
                "debug": False,
                "solver": "intensity",
            }

            stress_map = main.image_to_stress(params)

            # Check binned dimensions (should be half)
            assert stress_map.shape[0] == 3, "Binned height should be 3"
            assert stress_map.shape[1] == 3, "Binned width should be 3"
            assert stress_map.shape[2] == 3, "Should have 3 stress components"

    def test_image_to_stress_with_c_array(self):
        """Test image_to_stress with array of C values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(5, 5, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = {
                "input_filename": input_file,
                "C": [3e-9, 3.5e-9, 2.5e-9],  # Different C for each wavelength
                "thickness": 0.01,
                "wavelengths": [650, 550, 450],
                "S_i_hat": [1.0, 0.0, 0.0],
                "debug": False,
                "solver": "intensity",
            }

            stress_map = main.image_to_stress(params)
            assert stress_map.shape == (5, 5, 3), "Stress map shape should match"

    def test_image_to_stress_missing_parameters(self):
        """Test image_to_stress raises error when neither folderName nor input_filename provided."""
        params = {
            "C": 3e-9,
            "thickness": 0.01,
            "wavelengths": [650, 550, 450],
            "S_i_hat": [1.0, 0.0, 0.0],
            "debug": False,
        }

        with pytest.raises(ValueError, match="Either 'folderName' or 'input_filename' must be specified"):
            main.image_to_stress(params)

    def test_image_to_stress_with_n_jobs(self):
        """Test image_to_stress with custom n_jobs parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(2, 2, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = {
                "input_filename": input_file,
                "C": 3e-9,
                "thickness": 0.01,
                "wavelengths": [650, 550, 450],
                "S_i_hat": [1.0, 0.0, 0.0],
                "debug": False,
                "n_jobs": 1,  # Use single core
                "solver": "intensity",
            }

            stress_map = main.image_to_stress(params)
            assert stress_map.shape == (2, 2, 3), "Stress map should be computed"

    @patch("photoelastimetry.main.photoelastimetry.optimiser.equilibrium_mean_stress.recover_mean_stress")
    @patch("photoelastimetry.main.photoelastimetry.seeding.phase_decomposed_seeding")
    def test_image_to_stress_global_mean_stress_solver(self, mock_seeding, mock_recover_mean_stress):
        """Test image_to_stress with the global_mean_stress solver path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(2, 3, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            mock_seeding.return_value = np.ones((2, 3, 3), dtype=float)

            s_xx = np.full((2, 3), 1.0)
            s_yy = np.full((2, 3), 2.0)
            s_xy = np.full((2, 3), 0.5)
            fake_wrapper = MagicMock()
            fake_wrapper.get_stress_fields.return_value = (s_xx, s_yy, s_xy)
            mock_recover_mean_stress.return_value = (fake_wrapper, np.zeros(1))

            params = {
                "input_filename": input_file,
                "C": [3e-9, 3e-9, 3e-9],
                "thickness": 0.01,
                "wavelengths": [650, 550, 450],
                "S_i_hat": [1.0, 0.0, 0.0],
                "debug": False,
                "solver": "global_mean_stress",
                "global_mean_stress": {"max_iterations": 1, "verbose": False},
            }

            stress_map = main.image_to_stress(params)

            assert stress_map.shape == (2, 3, 3), "Stress map should have shape [H, W, 3]"
            assert np.allclose(stress_map[:, :, 0], s_xx)
            assert np.allclose(stress_map[:, :, 1], s_yy)
            assert np.allclose(stress_map[:, :, 2], s_xy)
            mock_recover_mean_stress.assert_called_once()

    @patch("photoelastimetry.main.photoelastimetry.optimiser.equilibrium_mean_stress.recover_mean_stress")
    @patch("photoelastimetry.main.photoelastimetry.seeding.phase_decomposed_seeding")
    def test_image_to_stress_default_solver_is_global_mean_stress(
        self, mock_seeding, mock_recover_mean_stress
    ):
        """If solver is omitted, image_to_stress should default to global_mean_stress."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(2, 3, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            mock_seeding.return_value = np.ones((2, 3, 3), dtype=float)
            fake_wrapper = MagicMock()
            fake_wrapper.get_stress_fields.return_value = (
                np.zeros((2, 3)),
                np.zeros((2, 3)),
                np.zeros((2, 3)),
            )
            mock_recover_mean_stress.return_value = (fake_wrapper, np.zeros(1))

            params = {
                "input_filename": input_file,
                "C": [3e-9, 3e-9, 3e-9],
                "thickness": 0.01,
                "wavelengths": [650, 550, 450],
                "S_i_hat": [1.0, 0.0, 0.0],
                "debug": False,
                # solver intentionally omitted
                "global_mean_stress": {"max_iterations": 1, "verbose": False},
            }

            stress_map = main.image_to_stress(params)
            assert stress_map.shape == (2, 3, 3)
            mock_recover_mean_stress.assert_called_once()


class TestStressToImage:
    """Tests for stress_to_image function."""

    def test_stress_to_image_saves_stack_tiff(self):
        """stress_to_image should generate and save a full [H, W, C, A] synthetic stack."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stress_data = np.random.rand(4, 5, 3).astype(np.float32) * 1e6
            stress_file = os.path.join(tmpdir, "stress.npy")
            io.save_image(stress_file, stress_data)
            output_file = os.path.join(tmpdir, "synthetic_stack.tiff")

            params = {
                "stress_filename": stress_file,
                "thickness": 0.01,
                "wavelengths": [650, 550, 450],
                "C": [3e-9, 3.5e-9, 2.5e-9],
                "S_i_hat": [1.0, 0.0, 0.0],
                "output_filename": output_file,
                "scattering": 0,
            }

            synthetic = main.stress_to_image(params)
            assert synthetic.shape == (4, 5, 3, 4)
            assert os.path.exists(output_file)

            loaded, _ = io.load_image(output_file)
            assert loaded.shape == (4, 5, 3, 4)

    def test_stress_to_image_with_legacy_stress_order(self):
        """Legacy [xy, yy, xx] ordering should still be supported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stress_xx_yy_xy = np.random.rand(3, 4, 3).astype(np.float32) * 1e6
            stress_xy_yy_xx = np.stack(
                [stress_xx_yy_xy[:, :, 2], stress_xx_yy_xy[:, :, 1], stress_xx_yy_xy[:, :, 0]],
                axis=-1,
            )

            standard_file = os.path.join(tmpdir, "standard.npy")
            legacy_file = os.path.join(tmpdir, "legacy.npy")
            io.save_image(standard_file, stress_xx_yy_xy)
            io.save_image(legacy_file, stress_xy_yy_xx)

            params_standard = {
                "stress_filename": standard_file,
                "thickness": 0.01,
                "lambda_light": 650e-9,
                "C": 3e-9,
                "output_filename": os.path.join(tmpdir, "standard.tiff"),
                "stress_order": "xx_yy_xy",
            }
            params_legacy = {
                "stress_filename": legacy_file,
                "thickness": 0.01,
                "lambda_light": 650e-9,
                "C": 3e-9,
                "output_filename": os.path.join(tmpdir, "legacy.tiff"),
                "stress_order": "xy_yy_xx",
            }

            synthetic_standard = main.stress_to_image(params_standard)
            synthetic_legacy = main.stress_to_image(params_legacy)
            assert np.allclose(synthetic_standard, synthetic_legacy, atol=1e-10, rtol=1e-6)

    def test_stress_to_image_default_output(self):
        """Default output filename should remain output.png for plot output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stress_data = np.random.rand(5, 5, 3).astype(np.float32) * 1e6
            stress_file = os.path.join(tmpdir, "stress.npy")
            io.save_image(stress_file, stress_data)

            params = {
                "stress_filename": stress_file,
                "scattering": 0,
                "thickness": 0.01,
                "lambda_light": 650e-9,
                "C": 3e-9,
                # No output_filename -> should default to output.png
            }

            with patch("photoelastimetry.plotting.plot_fringe_pattern") as mock_plot:
                synthetic = main.stress_to_image(params)
                assert synthetic.shape == (5, 5, 1, 4)
                mock_plot.assert_called_once()
                assert mock_plot.call_args[1]["filename"] == "output.png"


class TestDemosaicRawImage:
    """Tests for demosaic_raw_image function."""

    def test_demosaic_raw_image_tiff(self):
        """Test de-mosaicing raw image to TIFF format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic raw image (4x4 superpixel pattern)
            # For a 16x16 image, we get 4x4 demosaiced
            raw_data = np.random.randint(0, 4096, (16, 16), dtype=np.uint16)
            raw_file = os.path.join(tmpdir, "test.raw")
            raw_data.tofile(raw_file)

            metadata = {"width": 16, "height": 16, "dtype": "uint16"}

            output_prefix = os.path.join(tmpdir, "demosaiced")

            demosaiced = main.demosaic_raw_image(
                raw_file, metadata, output_prefix=output_prefix, output_format="tiff"
            )

            # Check demosaiced shape: [H/4, W/4, 3 (RGB), 4 (angles)]
            assert demosaiced.shape == (4, 4, 3, 4), "Demosaiced shape should be [H/4, W/4, 3, 4]"

            # Check that TIFF file was created
            tiff_file = f"{output_prefix}_demosaiced.tiff"
            assert os.path.exists(tiff_file), "TIFF output file should be created"

    def test_demosaic_raw_image_png(self):
        """Test de-mosaicing raw image to PNG format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_data = np.random.randint(0, 4096, (16, 16), dtype=np.uint16)
            raw_file = os.path.join(tmpdir, "test.raw")
            raw_data.tofile(raw_file)

            metadata = {"width": 16, "height": 16, "dtype": "uint16"}
            output_prefix = os.path.join(tmpdir, "demosaiced")

            demosaiced = main.demosaic_raw_image(
                raw_file, metadata, output_prefix=output_prefix, output_format="png"
            )

            # Check demosaiced shape
            assert demosaiced.shape == (4, 4, 3, 4), "Demosaiced shape should be correct"

            # Check that 4 PNG files were created (one per angle)
            angle_names = ["0deg", "45deg", "90deg", "135deg"]
            for angle in angle_names:
                png_file = f"{output_prefix}_{angle}.png"
                assert os.path.exists(png_file), f"PNG file for {angle} should be created"

    def test_demosaic_default_output_prefix(self):
        """Test de-mosaicing with default output prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_data = np.random.randint(0, 4096, (16, 16), dtype=np.uint16)
            raw_file = os.path.join(tmpdir, "test.raw")
            raw_data.tofile(raw_file)

            metadata = {"width": 16, "height": 16, "dtype": "uint16"}

            main.demosaic_raw_image(raw_file, metadata, output_format="tiff")

            # Check that output uses input filename as prefix
            expected_output = os.path.join(tmpdir, "test_demosaiced.tiff")
            assert os.path.exists(expected_output), "Default output filename should be used"

    def test_demosaic_invalid_format(self):
        """Test de-mosaicing with invalid output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_data = np.random.randint(0, 4096, (16, 16), dtype=np.uint16)
            raw_file = os.path.join(tmpdir, "test.raw")
            raw_data.tofile(raw_file)

            metadata = {"width": 16, "height": 16, "dtype": "uint16"}

            with pytest.raises(ValueError, match="Unsupported output format"):
                main.demosaic_raw_image(raw_file, metadata, output_format="invalid")


class TestCLIFunctions:
    """Tests for CLI functions."""

    @patch("photoelastimetry.main.image_to_stress")
    @patch("builtins.open", create=True)
    @patch("json5.load")
    @patch("argparse.ArgumentParser.parse_args")
    def test_cli_image_to_stress(self, mock_args, mock_json_load, mock_open, mock_image_to_stress):
        """Test CLI wrapper for image_to_stress."""
        # Mock command line arguments
        mock_args.return_value = MagicMock(json_filename="test.json5", output=None)

        # Mock file opening and JSON loading
        mock_params = {"test": "params"}
        mock_json_load.return_value = mock_params

        # Call CLI function
        main.cli_image_to_stress()

        # Verify that image_to_stress was called with correct params
        mock_image_to_stress.assert_called_once_with(mock_params, output_filename=None)

    @patch("photoelastimetry.main.stress_to_image")
    @patch("builtins.open", create=True)
    @patch("json5.load")
    @patch("argparse.ArgumentParser.parse_args")
    def test_cli_stress_to_image(self, mock_args, mock_json_load, mock_open, mock_stress_to_image):
        """Test CLI wrapper for stress_to_image."""
        mock_args.return_value = MagicMock(json_filename="test.json5")
        mock_params = {"test": "params"}
        mock_json_load.return_value = mock_params

        main.cli_stress_to_image()

        mock_stress_to_image.assert_called_once_with(mock_params)

    @patch("photoelastimetry.main.demosaic_raw_image")
    @patch("argparse.ArgumentParser.parse_args")
    def test_cli_demosaic_single_file(self, mock_args, mock_demosaic):
        """Test CLI demosaic for single file."""
        mock_args.return_value = MagicMock(
            input_file="test.raw",
            width=4096,
            height=3000,
            dtype=None,
            output_prefix=None,
            format="tiff",
            all=False,
        )

        main.cli_demosaic()

        # Verify demosaic was called once
        mock_demosaic.assert_called_once()


class TestIntegrationWithRealData:
    """Integration tests using the actual test data file if it exists."""

    def test_image_to_stress_with_test_json(self):
        """Test image_to_stress with the actual test.json5 config if available."""
        repo_root = Path(__file__).resolve().parents[1]
        json_file = repo_root / "json" / "test.json5"
        if not json_file.exists():
            pytest.skip("Test data file not available")

        # Load the test configuration
        with open(json_file, "r") as f:
            params = json5.load(f)

        # Check if input file exists
        if "input_filename" in params:
            input_path = repo_root / params["input_filename"]
            if not input_path.exists():
                pytest.skip("Test input image not available")

            # Update to absolute path
            params["input_filename"] = str(input_path)

            # Don't save output in test
            if "output_filename" in params:
                del params["output_filename"]

            # Run the function
            stress_map = main.image_to_stress(params)

            # Basic validation
            assert stress_map is not None, "Stress map should be generated"
            assert stress_map.ndim == 3, "Stress map should be 3D"
            assert np.isfinite(stress_map).any(), "Stress map should contain finite values"
