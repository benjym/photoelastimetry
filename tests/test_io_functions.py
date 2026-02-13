import json
from pathlib import Path

import numpy as np
import pytest

from photoelastimetry.io import bin_image, load_image, load_raw, read_raw, save_image, split_channels


def test_split_channels_known_superpixel_mapping():
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)
    out = split_channels(data)

    assert out.shape == (1, 1, 4, 4)
    # R channel values from the documented superpixel map.
    assert out[0, 0, 0, 0] == data[0, 0]  # R_0
    assert out[0, 0, 0, 1] == data[0, 1]  # R_45
    assert out[0, 0, 0, 2] == data[1, 1]  # R_90
    assert out[0, 0, 0, 3] == data[1, 0]  # R_135


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_read_raw_auto_detects_dtype(tmp_path, dtype):
    arr = np.arange(36, dtype=dtype).reshape(6, 6)
    raw_file = tmp_path / f"frame_{dtype.__name__}.raw"
    arr.tofile(raw_file)

    metadata = {"width": 6, "height": 6}
    loaded = read_raw(str(raw_file), metadata)

    assert metadata["dtype"] == np.dtype(dtype).name
    assert loaded.shape == (6, 6)
    assert np.array_equal(np.asarray(loaded), arr)


def test_read_raw_invalid_size_raises(tmp_path):
    raw_file = tmp_path / "bad.raw"
    raw_file.write_bytes(b"abc")

    with pytest.raises(ValueError, match="File size does not match expected size"):
        read_raw(str(raw_file), {"width": 10, "height": 10})


def test_load_raw_missing_metadata_raises(tmp_path):
    with pytest.raises(ValueError, match="recordingMetadata.json"):
        load_raw(str(tmp_path))


def test_load_raw_no_frames_raises(tmp_path):
    (tmp_path / "0000000").mkdir()
    (tmp_path / "recordingMetadata.json").write_text(json.dumps({"width": 4, "height": 4}))

    with pytest.raises(ValueError, match="No frames found"):
        load_raw(str(tmp_path))


def test_load_raw_median_and_split_channels(tmp_path):
    root = Path(tmp_path)
    frame_dir = root / "0000000"
    frame_dir.mkdir()

    (root / "recordingMetadata.json").write_text(json.dumps({"width": 4, "height": 4}))

    f0 = np.arange(16, dtype=np.uint8).reshape(4, 4)
    f1 = f0 + 2
    f0.tofile(frame_dir / "frame0000000000.raw")
    f1.tofile(frame_dir / "frame0000000001.raw")

    demosaiced, metadata = load_raw(str(root))

    expected_median = np.median(np.stack([f0, f1], axis=0), axis=0)
    expected = split_channels(expected_median)

    assert metadata["width"] == 4
    assert metadata["height"] == 4
    assert demosaiced.shape == (1, 1, 4, 4)
    assert np.allclose(demosaiced, expected)


def test_save_load_raw_round_trip(tmp_path):
    arr = np.arange(25, dtype=np.uint16).reshape(5, 5)
    raw_file = tmp_path / "test.raw"

    save_image(str(raw_file), arr, metadata={"dtype": "uint16"})
    loaded, meta = load_image(str(raw_file), metadata={"dtype": "uint16", "width": 5, "height": 5})

    assert loaded.dtype == np.uint16
    assert np.array_equal(loaded, arr)
    assert meta["width"] == 5
    assert meta["height"] == 5


def test_save_load_tiff_4d_preserves_shape_and_values(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(6, 7, 3, 4)).astype(np.float32)
    filename = tmp_path / "stack.tiff"

    save_image(str(filename), data)
    loaded, _ = load_image(str(filename))

    assert loaded.shape == data.shape
    assert np.allclose(loaded, data, rtol=1e-6, atol=1e-6)


def test_save_load_npy_round_trip(tmp_path):
    data = np.random.default_rng(1).normal(size=(10, 3)).astype(np.float64)
    filename = tmp_path / "arr.npy"

    save_image(str(filename), data)
    loaded, _ = load_image(str(filename))

    assert np.array_equal(loaded, data)


def test_bin_image_non_divisible_dimensions_truncates_edges():
    data = np.arange(7 * 10 * 2 * 4, dtype=float).reshape(7, 10, 2, 4)
    binned = bin_image(data, 3)

    # 7x10 with bin 3 => 2x3 bins after truncation.
    assert binned.shape == (2, 3, 2, 4)

    expected = data[:6, :9].reshape(2, 3, 3, 3, 2, 4).mean(axis=(1, 3))
    assert np.allclose(binned, expected)


def test_save_and_load_unsupported_format_raises(tmp_path):
    bad = tmp_path / "bad.xyz"
    with pytest.raises(ValueError, match="Unsupported file format"):
        save_image(str(bad), np.zeros((2, 2)))

    bad.write_text("x")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_image(str(bad))
