import numpy as np
import pytest

from photoelastimetry.correction import compute_disorder_correction, estimate_grain_encounters


def test_compute_disorder_correction_limits():
    N = 100.0

    # Perfectly aligned case (|m| -> 1)
    K_aligned = compute_disorder_correction(N, order_param=1.0)
    # Expected: 1 / sqrt(1/N + 1) -> approx 1
    assert np.isclose(K_aligned, 1 / np.sqrt(0.01 + 1))
    assert K_aligned < 1.0  # slightly less than 1 due to 1/N term

    # Random case (|m| -> 0)
    K_random = compute_disorder_correction(N, order_param=0.0)
    # Expected: 1 / sqrt(1/N) = sqrt(N) = 10
    assert np.isclose(K_random, np.sqrt(N))

    # Partial coherence
    K_partial = compute_disorder_correction(N, order_param=0.5)
    assert 1.0 < K_partial < np.sqrt(N)


def test_estimate_grain_encounters():
    # N = 1.5 * nu * H / d
    nu = 0.6
    L = 0.01  # 10 mm
    d = 0.001  # 1 mm

    # Expected: 1.5 * 0.6 * 10 = 9
    N = estimate_grain_encounters(nu, L, d)
    assert np.isclose(N, 9.0)


def test_zero_handling():
    assert compute_disorder_correction(0, 0.5) == 1.0
    assert estimate_grain_encounters(0.6, 0.01, 0) == 1.0
