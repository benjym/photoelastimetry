import numpy as np
import pytest

from photoelastimetry.visualisation import ascii_boundary_conditions


@pytest.mark.smoke
def test_ascii_boundary_conditions_smoke():
    h, w = 12, 16
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True

    vals = {
        "xx": np.full((h, w), np.nan),
        "yy": np.full((h, w), np.nan),
        "xy": np.full((h, w), np.nan),
    }
    vals["yy"][0, :] = 0.0
    vals["xy"][0, :] = 0.0
    vals["xx"][:, 0] = 10.0

    txt = ascii_boundary_conditions(mask, vals, title="Smoke Boundary", downsample_factor=2)

    assert "Smoke Boundary" in txt
    assert "Legend" in txt
    assert len(txt.splitlines()) > 6
