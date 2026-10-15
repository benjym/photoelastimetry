"""Forward-model round trips and file/CLI checks for quick optical inversion."""

import sys

import numpy as np
import pytest
from matplotlib.image import imread

from photoelastimetry.image import mueller_matrix
from photoelastimetry.io import load_image, save_image
from photoelastimetry.quick import cli_image_to_retardation, first_period_inversion, image_to_retardation


def intensities(theta, delta, source):
    measured = np.einsum("...ij,j->...i", mueller_matrix(theta, delta), [1, *source])
    return np.stack(
        [
            (1 + measured[..., 1]) / 2,
            (1 + measured[..., 2]) / 2,
            (1 - measured[..., 1]) / 2,
            (1 - measured[..., 2]) / 2,
        ],
        axis=-1,
    )


@pytest.mark.parametrize(
    "source,limit", [([1, 0, 0], np.pi), ([0, 1, 0], np.pi), ([0, 0, 1], np.pi / 2), ([0, 0, -1], np.pi / 2)]
)
def test_forward_round_trip(source, limit):
    theta = np.array([-0.42, 0.32, 0.64])[None, None, :]
    expected = np.array([0.15, 0.4, 0.85])[None, None, :] * limit
    delta, direction = first_period_inversion(intensities(theta, expected, source), source)
    np.testing.assert_allclose(delta, expected, atol=1e-10)
    period = np.pi if source[2] else np.pi / 2
    np.testing.assert_allclose(
        np.exp(2j * np.pi * direction / period), np.exp(2j * np.pi * theta / period), atol=1e-10
    )


@pytest.mark.parametrize("use_bright,use_dark", [(True, True), (True, False), (False, True)])
def test_fields_recover_optics(use_bright, use_dark):
    ideal = intensities(np.full((2, 3, 1), 0.3), np.full((2, 3, 1), 0.6), [0, 0, 1])
    gain = np.array([2.0, 4.0, 3.0, 6.0]) if use_bright else np.ones(4)
    dark = np.full_like(ideal, 12.0) if use_dark else np.zeros_like(ideal)
    delta, theta = first_period_inversion(
        ideal * gain + dark,
        [0, 0, 1],
        bright_field=gain * np.ones_like(ideal) + dark if use_bright else None,
        dark_field=dark if use_dark else None,
    )
    np.testing.assert_allclose(delta, 0.6)
    np.testing.assert_allclose(theta, 0.3)


def test_invalid_pixels_and_isoclinics():
    data = intensities(np.full((1, 5, 1), 0.3), np.full((1, 5, 1), 0.6), [1, 0, 0])
    data[0, 0] = 0
    data[0, 1, 0, 0] = np.nan
    data[0, 2, 0, 0] = -1
    data[0, 3] = intensities(0.0, 0.6, [1, 0, 0])
    delta, theta = first_period_inversion(data)
    assert np.isnan(delta[0, :4]).all()
    assert np.isnan(theta[0, :4]).all()
    assert np.isfinite(delta[0, 4]).all()
    bright = np.ones_like(data)
    bright[0, 4, 0, 1] = 0
    delta, _ = first_period_inversion(data, bright_field=bright)
    assert np.isnan(delta).all()


@pytest.mark.parametrize("source", [[1, 0], [0, 0, 0], [0, 0, 0.5], [0.6, 0, 0.8], [np.nan, 0, 0]])
def test_bad_source(source):
    with pytest.raises(ValueError):
        first_period_inversion(np.ones((2, 3, 1, 4)), source)


def test_circular_zero_retardation_has_undefined_direction():
    data = np.full((2, 3, 1, 4), 0.5)
    delta, theta = first_period_inversion(data, [0, 0, 1])
    np.testing.assert_array_equal(delta, 0)
    assert np.isnan(theta).all()


def test_bad_layout_and_references():
    with pytest.raises(ValueError, match="Expected"):
        first_period_inversion(np.ones((4, 5, 3)))
    for name in ("bright_field", "dark_field"):
        with pytest.raises(ValueError, match="match the input shape"):
            first_period_inversion(np.ones((4, 5, 1, 4)), **{name: np.ones((2, 3, 1, 4))})


def test_cli_tiff_and_png(tmp_path, monkeypatch):
    shape = (2, 5, 3)
    ideal = intensities(np.full(shape, 0.4), np.full(shape, 0.7), [0, 0, 1])
    save_image(str(tmp_path / "sample.tiff"), ideal * 100 + 10)
    save_image(str(tmp_path / "bright.tiff"), np.full_like(ideal, 110))
    save_image(str(tmp_path / "dark.tiff"), np.full_like(ideal, 10))
    prefix = tmp_path / "output" / "map"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "image-to-retardation",
            str(tmp_path / "sample.tiff"),
            "-o",
            str(prefix),
            "--input-stokes",
            "0",
            "0",
            "1",
            "--bright-field",
            str(tmp_path / "bright.tiff"),
            "--dark-field",
            str(tmp_path / "dark.tiff"),
        ],
    )
    cli_image_to_retardation()
    for name, expected in [("retardation", 0.7), ("direction", 0.4)]:
        values, _ = load_image(f"{prefix}_{name}.tiff")
        assert values.dtype == np.float32
        assert values.shape == shape
        np.testing.assert_allclose(values, expected, atol=1e-6)
    delta, theta = image_to_retardation(
        tmp_path / "sample.tiff", prefix, "png", [0, 0, 1], tmp_path / "bright.tiff", tmp_path / "dark.tiff"
    )
    for channel in range(3):
        for name, expected in [("retardation", 0.7 / (np.pi / 2)), ("direction", (0.4 + np.pi / 2) / np.pi)]:
            pixels = imread(f"{prefix}_{name}_c{channel}.png")
            np.testing.assert_allclose(pixels[..., 0], expected, atol=1 / 255)
    assert delta.shape == theta.shape == shape


def test_monochrome_and_raw_mosaic(tmp_path):
    mono = intensities(np.full((2, 3), 0.3), np.full((2, 3), 0.6), [0, 0, 1])
    delta, _ = first_period_inversion(mono, [0, 0, 1])
    np.testing.assert_allclose(delta, 0.6)
    mosaic = np.zeros((8, 12))
    for row, col in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        for dr, dc, angle in [(0, 0, 0), (1, 0, 1), (1, 1, 2), (0, 1, 3)]:
            mosaic[row + dr :: 4, col + dc :: 4] = mono[..., angle]
    save_image(str(tmp_path / "raw.tiff"), mosaic)
    result, _ = image_to_retardation(tmp_path / "raw.tiff", input_stokes=[0, 0, 1], raw_mosaic=True)
    np.testing.assert_allclose(result, 0.6, atol=1e-6)
