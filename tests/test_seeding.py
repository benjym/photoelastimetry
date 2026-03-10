"""
Tests for phase decomposed seeding method.
"""

import numpy as np

from photoelastimetry.image import predict_stokes
from photoelastimetry.seeding import (
    PhaseDecomposedSeed,
    invert_wrapped_retardance,
    phase_decomposed_seeding,
    resolve_fringe_orders,
    retardance_to_delta_sigma,
)

# Params
WAVELENGTHS = np.array([650e-9, 550e-9, 450e-9])
C_VALUES = np.array([2e-12, 2e-12, 2e-12])
NU = 1.0
L = 0.01
S_I_HAT = np.array([1.0, 0.0, 0.0])  # Linearly polarised light at 0 degrees


def _synthetic_stack_from_stress(sigma_xx, sigma_yy, sigma_xy, height=1, width=1):
    stokes = np.stack(
        [
            predict_stokes(sigma_xx, sigma_yy, sigma_xy, C_VALUES[i], NU, L, WAVELENGTHS[i], S_I_HAT)
            for i in range(len(WAVELENGTHS))
        ],
        axis=0,
    )

    i0 = (1 + stokes[:, 0]) / 2
    i90 = (1 - stokes[:, 0]) / 2
    i45 = (1 + stokes[:, 1]) / 2
    i135 = (1 - stokes[:, 1]) / 2

    data = np.zeros((height, width, len(WAVELENGTHS), 4), dtype=float)
    for i in range(len(WAVELENGTHS)):
        data[:, :, i, 0] = i0[i]
        data[:, :, i, 1] = i45[i]
        data[:, :, i, 2] = i90[i]
        data[:, :, i, 3] = i135[i]
    return data


def test_invert_wrapped_retardance():
    # Simulate a pixel with known stress
    sigma_diff = 1e6  # 1 MPa
    theta_true = np.pi / 6  # 30 deg

    s_xx = sigma_diff / 2 * (1 + np.cos(2 * theta_true))
    s_yy = sigma_diff / 2 * (1 - np.cos(2 * theta_true))
    s_xy = sigma_diff / 2 * np.sin(2 * theta_true)

    S_m_hat = np.stack(
        [
            predict_stokes(s_xx, s_yy, s_xy, C_VALUES[i], NU, L, WAVELENGTHS[i], S_I_HAT)
            for i in range(len(WAVELENGTHS))
        ],
        axis=0,
    )

    theta_rec, delta_wrap_rec = invert_wrapped_retardance(S_m_hat, S_I_HAT)

    # Orientation ambiguity: theta or theta - pi/2
    diff = np.abs(theta_rec - theta_true)
    diff2 = np.abs(theta_rec - (theta_true - np.pi / 2))
    assert diff < 0.15 or diff2 < 0.15


def test_resolve_fringe_orders():
    # Test multi-fringe resolution
    sigma_diff = 4e6  # 4 MPa -> should be > 1 fringe for some wavelengths

    # Calculate true deltas
    deltas = (2 * np.pi * C_VALUES * NU * L / WAVELENGTHS) * sigma_diff
    # Simulate wrapped/folded measurement
    # We simulate what invert_wrapped_retardance produces: 2*asin(|sin(delta/2)|)
    delta_folded = 2 * np.arcsin(np.abs(np.sin(deltas / 2)))

    rec_sigma = resolve_fringe_orders(delta_folded, WAVELENGTHS, C_VALUES, NU, L, sigma_max=10e6, n_max=6)

    print(f"True: {sigma_diff}, Rec: {rec_sigma}")
    assert np.isclose(rec_sigma, sigma_diff, rtol=0.1)


def test_phase_decomposed_seeding_returns_seed_result():
    sigma_diff = 4e6
    theta_true = np.pi / 6

    sigma_xx = sigma_diff / 2 * (1 + np.cos(2 * theta_true))
    sigma_yy = sigma_diff / 2 * (1 - np.cos(2 * theta_true))
    sigma_xy = sigma_diff / 2 * np.sin(2 * theta_true)
    data = _synthetic_stack_from_stress(sigma_xx, sigma_yy, sigma_xy, height=2, width=3)

    seed = phase_decomposed_seeding(data, WAVELENGTHS, C_VALUES, NU, L, S_i_hat=S_I_HAT)

    assert isinstance(seed, PhaseDecomposedSeed)
    assert seed.retardance.shape == (2, 3, 3)
    assert seed.theta.shape == (2, 3)
    assert seed.delta_sigma.shape == (2, 3)
    assert np.allclose(
        retardance_to_delta_sigma(seed.retardance, WAVELENGTHS, C_VALUES, NU, L), seed.delta_sigma
    )
    assert np.allclose(seed.delta_sigma, sigma_diff, rtol=0.1)
    assert np.allclose(np.cos(2 * seed.theta), np.cos(2 * theta_true), atol=0.15)
    assert np.allclose(np.sin(2 * seed.theta), np.sin(2 * theta_true), atol=0.15)


def test_seed_to_stress_map_matches_principal_invariant_formula():
    delta_sigma = np.array([[1.0e6, 2.0e6], [3.0e6, 4.0e6]])
    theta = np.array([[0.1, 0.2], [0.3, 0.4]])
    retardance = (2 * np.pi * C_VALUES * NU * L / WAVELENGTHS) * delta_sigma[..., np.newaxis]

    seed = PhaseDecomposedSeed(retardance=retardance, theta=theta, delta_sigma=delta_sigma)

    stress_map = seed.to_stress_map(K=0.25)
    sigma_1 = delta_sigma / (1 - 0.25)
    sigma_2 = 0.25 * sigma_1
    pressure = (sigma_1 + sigma_2) / 2
    expected = np.stack(
        [
            pressure + 0.5 * delta_sigma * np.cos(2 * theta),
            pressure - 0.5 * delta_sigma * np.cos(2 * theta),
            0.5 * delta_sigma * np.sin(2 * theta),
        ],
        axis=-1,
    )

    assert stress_map.shape == (2, 2, 3)
    assert np.allclose(stress_map, expected)
