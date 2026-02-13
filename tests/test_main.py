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

    @staticmethod
    def _base_params(input_file):
        return {
            "input_filename": input_file,
            "C": [3e-9, 3e-9, 3e-9],
            "thickness": 0.01,
            "wavelengths": [650, 550, 450],
            "S_i_hat": [1.0, 0.0, 0.0],
            "debug": False,
        }

    @pytest.fixture(autouse=True)
    def _mock_optimise_pipeline(self):
        with (
            patch("photoelastimetry.main.photoelastimetry.seeding.phase_decomposed_seeding") as mock_seeding,
            patch("photoelastimetry.main.photoelastimetry.optimise.recover_mean_stress") as mock_recover,
        ):

            def fake_seeding(image_stack, *_args, **_kwargs):
                h, w = image_stack.shape[:2]
                return np.ones((h, w, 3), dtype=float)

            def fake_recover(delta_sigma_map, theta_map, **_kwargs):
                h, w = delta_sigma_map.shape
                fake_wrapper = MagicMock()
                fake_wrapper.get_stress_fields.return_value = (
                    np.full((h, w), 1.0),
                    np.full((h, w), 2.0),
                    np.full((h, w), 0.5),
                )
                return fake_wrapper, np.zeros(1)

            mock_seeding.side_effect = fake_seeding
            mock_recover.side_effect = fake_recover
            yield mock_seeding, mock_recover

    def test_image_to_stress_with_input_filename(self):
        """Test image_to_stress with input_filename parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(1, 2, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            stress_map = main.image_to_stress(self._base_params(input_file))

            assert stress_map.ndim == 3, "Stress map should be 3D"
            assert stress_map.shape[0] == 1, "Height should match input"
            assert stress_map.shape[1] == 2, "Width should match input"
            assert stress_map.shape[2] == 3, "Should have 3 stress components"

    def test_image_to_stress_with_output_filename(self):
        """Test image_to_stress saves output when output_filename is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(2, 2, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            output_file = os.path.join(tmpdir, "test_output.tiff")
            io.save_image(input_file, data)

            params = self._base_params(input_file)
            params["output_filename"] = output_file
            params["S_i_hat"] = [1.0, 0.0]
            main.image_to_stress(params)

            assert os.path.exists(output_file), "Output file should be created"
            loaded, _ = io.load_image(output_file)
            assert loaded.shape == (2, 2, 3), "Output should be 3D stress tensor"

    def test_image_to_stress_with_crop(self):
        """Test image_to_stress with crop parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(20, 20, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = self._base_params(input_file)
            params["crop"] = [5, 7, 5, 7]
            stress_map = main.image_to_stress(params)

            assert stress_map.shape[0] == 2, "Cropped height should be 2"
            assert stress_map.shape[1] == 2, "Cropped width should be 2"
            assert stress_map.shape[2] == 3, "Should have 3 stress components"

    def test_image_to_stress_with_binning(self):
        """Test image_to_stress with binning parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(12, 12, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = self._base_params(input_file)
            params["binning"] = 4
            stress_map = main.image_to_stress(params)

            assert stress_map.shape[0] == 3, "Binned height should be 3"
            assert stress_map.shape[1] == 3, "Binned width should be 3"
            assert stress_map.shape[2] == 3, "Should have 3 stress components"

    def test_image_to_stress_with_c_array(self):
        """Test image_to_stress with array of C values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(5, 5, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = self._base_params(input_file)
            params["C"] = [3e-9, 3.5e-9, 2.5e-9]
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

    def test_image_to_stress_with_n_jobs_is_ignored(self):
        """n_jobs key should be ignored now that only optimise solver exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(2, 2, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = self._base_params(input_file)
            params["n_jobs"] = 1
            stress_map = main.image_to_stress(params)
            assert stress_map.shape == (2, 2, 3)

    def test_image_to_stress_uses_flat_optimise_options(self, _mock_optimise_pipeline):
        """Top-level optimise options should be passed to recover_mean_stress."""
        _mock_seeding, mock_recover = _mock_optimise_pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(2, 3, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = self._base_params(input_file)
            params["max_iterations"] = 1
            params["regularization_weight"] = 0.25
            params["boundary_weight"] = 2.0

            main.image_to_stress(params)
            kwargs = mock_recover.call_args.kwargs
            assert kwargs["max_iterations"] == 1
            assert kwargs["regularisation_weight"] == 0.25
            assert kwargs["boundary_weight"] == 2.0

    def test_image_to_stress_rejects_solver_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(2, 2, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)
            params = self._base_params(input_file)
            params["solver"] = "global_mean_stress"
            with pytest.raises(ValueError, match="`solver` is no longer supported"):
                main.image_to_stress(params)

    def test_image_to_stress_rejects_global_mean_stress_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(2, 2, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)
            params = self._base_params(input_file)
            params["global_mean_stress"] = {"max_iterations": 1}
            with pytest.raises(ValueError, match="Nested solver config blocks"):
                main.image_to_stress(params)

    def test_image_to_stress_rejects_global_solver_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(2, 2, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)
            params = self._base_params(input_file)
            params["global_solver"] = {"max_iterations": 1}
            with pytest.raises(ValueError, match="Nested solver config blocks"):
                main.image_to_stress(params)

    def test_image_to_stress_rejects_missing_boundary_mask_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(3, 4, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = self._base_params(input_file)
            params["boundary_mask_file"] = os.path.join(tmpdir, "missing_mask.npy")
            with pytest.raises(ValueError, match="Boundary mask file not found"):
                main.image_to_stress(params)

    def test_image_to_stress_rejects_boundary_mask_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(3, 4, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            mask_file = os.path.join(tmpdir, "mask.npy")
            io.save_image(input_file, data)
            io.save_image(mask_file, np.ones((2, 2), dtype=np.uint8))

            params = self._base_params(input_file)
            params["boundary_mask_file"] = mask_file
            with pytest.raises(ValueError, match="Boundary mask shape must be"):
                main.image_to_stress(params)

    def test_image_to_stress_rejects_missing_boundary_value_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(3, 4, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = self._base_params(input_file)
            params["boundary_values_files"] = {"xx": os.path.join(tmpdir, "missing_xx.npy")}
            with pytest.raises(ValueError, match="Boundary values file for 'xx' not found"):
                main.image_to_stress(params)

    def test_image_to_stress_rejects_boundary_value_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(3, 4, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            bad_xx = os.path.join(tmpdir, "bad_xx.npy")
            io.save_image(input_file, data)
            io.save_image(bad_xx, np.ones((3, 3), dtype=float))

            params = self._base_params(input_file)
            params["boundary_values_files"] = {"xx": bad_xx}
            with pytest.raises(ValueError, match="Boundary values 'xx' shape must be"):
                main.image_to_stress(params)

    def test_image_to_stress_rejects_external_potential_conflict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(3, 4, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            potential_file = os.path.join(tmpdir, "potential.npy")
            io.save_image(input_file, data)
            io.save_image(potential_file, np.zeros((3, 4), dtype=float))

            params = self._base_params(input_file)
            params["external_potential_file"] = potential_file
            params["external_potential_gradient"] = [1.0, -0.5]
            with pytest.raises(
                ValueError, match="Use either external_potential_file or external_potential_gradient"
            ):
                main.image_to_stress(params)

    def test_image_to_stress_rejects_external_potential_file_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(3, 4, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            potential_file = os.path.join(tmpdir, "potential.npy")
            io.save_image(input_file, data)
            io.save_image(potential_file, np.zeros((2, 2), dtype=float))

            params = self._base_params(input_file)
            params["external_potential_file"] = potential_file
            with pytest.raises(ValueError, match="External potential shape must be"):
                main.image_to_stress(params)

    def test_image_to_stress_rejects_invalid_external_gradient_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(3, 4, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = self._base_params(input_file)
            params["external_potential_gradient"] = [1.0, 2.0, 3.0]
            with pytest.raises(ValueError, match="external_potential_gradient must be"):
                main.image_to_stress(params)

    def test_image_to_stress_external_gradient_constructs_and_adds_potential(self, _mock_optimise_pipeline):
        _mock_seeding, mock_recover = _mock_optimise_pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(3, 4, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "test_input.tiff")
            io.save_image(input_file, data)

            params = self._base_params(input_file)
            params["external_potential_gradient"] = [2.0, -1.0]
            stress_map = main.image_to_stress(params)

            ext = mock_recover.call_args.kwargs["external_potential"]
            y_idx, x_idx = np.indices((3, 4), dtype=float)
            expected = 2.0 * x_idx - 1.0 * y_idx
            assert np.allclose(ext, expected)
            assert np.allclose(stress_map[:, :, 0], 1.0 + expected)
            assert np.allclose(stress_map[:, :, 1], 2.0 + expected)
            assert np.allclose(stress_map[:, :, 2], 0.5)

    def test_image_to_stress_uses_calibration_file_for_c_si_hat_and_blank_correction(
        self, _mock_optimise_pipeline
    ):
        mock_seeding, _mock_recover = _mock_optimise_pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.full((3, 3, 3, 4), 1.0, dtype=np.float32)
            input_file = os.path.join(tmpdir, "input.tiff")
            io.save_image(input_file, data)

            calibration_file = os.path.join(tmpdir, "calibration.json5")
            profile = {
                "version": 1,
                "method": "brazilian_disk",
                "wavelengths": [650e-9, 550e-9, 450e-9],
                "C": [9e-9, 8e-9, 7e-9],
                "S_i_hat": [0.2, 0.1, 0.95],
                "blank_correction": {
                    "offset": [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
                    "scale": [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
                    "mode": "dark_blank_scalar",
                },
                "fit_metrics": {},
                "provenance": {},
            }
            with open(calibration_file, "w") as f:
                json5.dump(profile, f)

            params = {
                "input_filename": input_file,
                "thickness": 0.01,
                "debug": False,
                "calibration_file": calibration_file,
            }

            main.image_to_stress(params)

            call = mock_seeding.call_args
            corrected_data = call.args[0]
            c_values = call.args[2]
            s_i_hat = call.kwargs["S_i_hat"]

            assert np.allclose(corrected_data, 1.8)  # (1.0 - 0.1) * 2.0
            assert np.allclose(c_values, profile["C"])
            assert np.allclose(s_i_hat, profile["S_i_hat"])

    def test_image_to_stress_explicit_params_override_calibration_profile(self, _mock_optimise_pipeline):
        mock_seeding, _mock_recover = _mock_optimise_pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(3, 3, 3, 4).astype(np.float32)
            input_file = os.path.join(tmpdir, "input.tiff")
            io.save_image(input_file, data)

            calibration_file = os.path.join(tmpdir, "calibration.json5")
            with open(calibration_file, "w") as f:
                json5.dump(
                    {
                        "version": 1,
                        "method": "brazilian_disk",
                        "wavelengths": [650e-9, 550e-9, 450e-9],
                        "C": [9e-9, 8e-9, 7e-9],
                        "S_i_hat": [0.2, 0.1, 0.95],
                        "blank_correction": {
                            "offset": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                            "scale": [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                            "mode": "identity",
                        },
                        "fit_metrics": {},
                        "provenance": {},
                    },
                    f,
                )

            params = self._base_params(input_file)
            params["C"] = [1e-9, 2e-9, 3e-9]
            params["S_i_hat"] = [1.0, 0.0, 0.0]
            params["calibration_file"] = calibration_file

            main.image_to_stress(params)
            call = mock_seeding.call_args

            assert np.allclose(call.args[2], [1e-9, 2e-9, 3e-9])
            assert np.allclose(call.kwargs["S_i_hat"], [1.0, 0.0, 0.0])


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

    def test_stress_to_image_rejects_missing_stress_filename(self):
        with pytest.raises(ValueError, match="Missing stress map path"):
            main.stress_to_image({"thickness": 0.01, "lambda_light": 650e-9, "C": 3e-9})

    def test_stress_to_image_rejects_missing_wavelengths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stress = np.random.rand(3, 3, 3).astype(np.float32)
            stress_file = os.path.join(tmpdir, "stress.npy")
            io.save_image(stress_file, stress)

            with pytest.raises(ValueError, match="Missing wavelengths"):
                main.stress_to_image({"stress_filename": stress_file, "thickness": 0.01, "C": 3e-9})

    def test_stress_to_image_rejects_invalid_c_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stress = np.random.rand(3, 3, 3).astype(np.float32)
            stress_file = os.path.join(tmpdir, "stress.npy")
            io.save_image(stress_file, stress)

            params = {
                "stress_filename": stress_file,
                "thickness": 0.01,
                "wavelengths": [650e-9, 550e-9, 450e-9],
                "C": [3e-9, 3e-9],  # wrong length
            }
            with pytest.raises(ValueError, match="C must be scalar or length 3"):
                main.stress_to_image(params)

    def test_stress_to_image_rejects_invalid_s_i_hat_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stress = np.random.rand(3, 3, 3).astype(np.float32)
            stress_file = os.path.join(tmpdir, "stress.npy")
            io.save_image(stress_file, stress)

            params = {
                "stress_filename": stress_file,
                "thickness": 0.01,
                "lambda_light": 650e-9,
                "C": 3e-9,
                "S_i_hat": [1.0],
            }
            with pytest.raises(ValueError, match="S_i_hat must have length 2 or 3"):
                main.stress_to_image(params)

    def test_stress_to_image_rejects_unsupported_output_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stress = np.random.rand(3, 3, 3).astype(np.float32)
            stress_file = os.path.join(tmpdir, "stress.npy")
            io.save_image(stress_file, stress)

            params = {
                "stress_filename": stress_file,
                "thickness": 0.01,
                "lambda_light": 650e-9,
                "C": 3e-9,
                "output_filename": os.path.join(tmpdir, "out.gif"),
            }
            with pytest.raises(ValueError, match="Unsupported output extension"):
                main.stress_to_image(params)

    def test_stress_to_image_uses_fallback_param_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stress = np.random.rand(3, 3, 3).astype(np.float32)
            stress_file = os.path.join(tmpdir, "stress.npy")
            io.save_image(stress_file, stress)

            param_file = os.path.join(tmpdir, "params.json5")
            output_file = os.path.join(tmpdir, "synthetic.npy")
            with open(param_file, "w") as f:
                json5.dump(
                    {
                        "stress_filename": stress_file,
                        "thickness": 0.01,
                        "wavelengths": [650e-9, 550e-9, 450e-9],
                        "C": [3e-9, 3e-9, 3e-9],
                        "output_filename": output_file,
                    },
                    f,
                )

            synthetic = main.stress_to_image({"p_filename": param_file})
            assert synthetic.shape == (3, 3, 3, 4)
            assert os.path.exists(output_file)

    def test_stress_to_image_uses_calibration_file_for_missing_c_and_s_i_hat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stress = np.random.rand(3, 3, 3).astype(np.float32)
            stress_file = os.path.join(tmpdir, "stress.npy")
            io.save_image(stress_file, stress)

            calibration_file = os.path.join(tmpdir, "calibration.json5")
            with open(calibration_file, "w") as f:
                json5.dump(
                    {
                        "version": 1,
                        "method": "brazilian_disk",
                        "wavelengths": [650e-9, 550e-9, 450e-9],
                        "C": [3e-9, 3e-9, 3e-9],
                        "S_i_hat": [0.9, 0.1, 0.0],
                        "blank_correction": {
                            "offset": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                            "scale": [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                            "mode": "identity",
                        },
                        "fit_metrics": {},
                        "provenance": {},
                    },
                    f,
                )

            params = {
                "stress_filename": stress_file,
                "thickness": 0.01,
                "calibration_file": calibration_file,
                "output_filename": os.path.join(tmpdir, "synthetic.tiff"),
            }
            synthetic = main.stress_to_image(params)
            assert synthetic.shape == (3, 3, 3, 4)

    def test_stress_to_image_explicit_params_override_calibration_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stress = np.random.rand(2, 2, 3).astype(np.float32)
            stress_file = os.path.join(tmpdir, "stress.npy")
            io.save_image(stress_file, stress)

            calibration_file = os.path.join(tmpdir, "calibration.json5")
            with open(calibration_file, "w") as f:
                json5.dump(
                    {
                        "version": 1,
                        "method": "brazilian_disk",
                        "wavelengths": [650e-9],
                        "C": [3e-9],
                        "S_i_hat": [0.2, 0.3, 0.4],
                        "blank_correction": {
                            "offset": [[0.0, 0.0, 0.0, 0.0]],
                            "scale": [[1.0, 1.0, 1.0, 1.0]],
                            "mode": "identity",
                        },
                        "fit_metrics": {},
                        "provenance": {},
                    },
                    f,
                )

            with patch("photoelastimetry.main.simulate_four_step_polarimetry") as mock_sim:
                zeros = np.zeros((2, 2), dtype=float)
                mock_sim.return_value = (zeros, zeros, zeros, zeros)

                params = {
                    "stress_filename": stress_file,
                    "thickness": 0.01,
                    "wavelengths": [650e-9],
                    "C": [7e-9],
                    "S_i_hat": [1.0, 0.0, 0.0],
                    "calibration_file": calibration_file,
                    "output_filename": os.path.join(tmpdir, "synthetic.tiff"),
                }
                main.stress_to_image(params)

                sim_call = mock_sim.call_args
                assert np.isclose(sim_call.args[3], 7e-9)
                assert np.allclose(sim_call.args[7], [1.0, 0.0, 0.0])


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

    @patch("photoelastimetry.main.photoelastimetry.calibrate.run_calibration")
    @patch("builtins.open", create=True)
    @patch("json5.load")
    @patch("argparse.ArgumentParser.parse_args")
    def test_cli_calibrate(self, mock_args, mock_json_load, mock_open, mock_run_calibration):
        mock_args.return_value = MagicMock(json_filename="calibration.json5")
        mock_params = {"method": "brazilian_disk"}
        mock_json_load.return_value = mock_params
        mock_run_calibration.return_value = {
            "profile_file": "calibration_profile.json5",
            "report_file": "calibration_report.md",
            "diagnostics_file": "calibration_diagnostics.npz",
        }

        with patch("builtins.print") as mock_print:
            main.cli_calibrate()

        mock_run_calibration.assert_called_once_with(mock_params)
        assert mock_print.call_count == 3

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

    @patch("photoelastimetry.main.demosaic_raw_image")
    @patch("glob.glob")
    @patch("tqdm.tqdm", side_effect=lambda seq, desc=None: seq)
    @patch("argparse.ArgumentParser.parse_args")
    def test_cli_demosaic_all_processes_each_file(self, mock_args, _mock_tqdm, mock_glob, mock_demosaic):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = os.path.join(tmpdir, "a.raw")
            file_b = os.path.join(tmpdir, "nested", "b.raw")
            os.makedirs(os.path.dirname(file_b), exist_ok=True)
            Path(file_a).write_bytes(b"")
            Path(file_b).write_bytes(b"")

            mock_args.return_value = MagicMock(
                input_file=tmpdir,
                width=8,
                height=8,
                dtype="uint16",
                output_prefix=None,
                format="tiff",
                all=True,
            )
            mock_glob.return_value = [file_a, file_b]

            main.cli_demosaic()

            assert mock_demosaic.call_count == 2

    @patch("argparse.ArgumentParser.parse_args")
    def test_cli_demosaic_all_requires_directory(self, mock_args):
        mock_args.return_value = MagicMock(
            input_file="not_a_directory.raw",
            width=8,
            height=8,
            dtype=None,
            output_prefix=None,
            format="tiff",
            all=True,
        )

        with pytest.raises(ValueError, match="input_file must be a directory"):
            main.cli_demosaic()


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
