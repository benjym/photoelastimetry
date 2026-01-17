"""
Tests for phase decomposed seeding method.
"""

import numpy as np
import pytest

from photoelastimetry.optimiser.stokes import predict_stokes
from photoelastimetry.seeding import invert_wrapped_retardance, resolve_fringe_orders

# Params
WAVELENGTHS = np.array([650e-9, 550e-9, 450e-9])
C_VALUES = np.array([2e-12, 2e-12, 2e-12])
NU = 1.0
L = 0.01
S_I_HAT = np.array([1.0, 0.0])


def test_invert_wrapped_retardance():
    # Simulate a pixel with known stress
    sigma_diff = 1e6  # 1 MPa
    theta_true = np.pi / 6  # 30 deg

    s_xx = sigma_diff / 2 * (1 + np.cos(2 * theta_true))
    s_yy = sigma_diff / 2 * (1 - np.cos(2 * theta_true))
    s_xy = sigma_diff / 2 * np.sin(2 * theta_true)

    # Predict Stokes
    # We need to broadcast wavelengths to (3,)
    # predict_stokes returns (S1, S2) for scalar wavelength?
    # Or (N, 2) for array wavelength?
    # Let's check predict_stokes signature. It takes scalar or array wavelength.
    # If we pass array, it returns array.

    S_m_hat = predict_stokes(s_xx, s_yy, s_xy, C_VALUES, NU, L, WAVELENGTHS, S_I_HAT)
    S_m_hat = S_m_hat.T  # predict_stokes usually returns [2, N] or similar?
    # Actually predict_stokes in stokes.py:
    # returns S1_hat, S2_hat.
    # If wavelength is array, return is likely broadcasted.
    # Let's assume predict_stokes returns (N_wl, 2) or (2, N_wl).
    # Checking stokes.py:
    # delta = ...
    # S1_hat = ...
    # return np.array([S1_hat, S2_hat])
    # So if delta is array, return is (2, N_wl).

    if S_m_hat.shape[0] == 2:
        S_m_hat = S_m_hat.T  # (N_wl, 2)

    theta_rec, delta_wrap_rec = invert_wrapped_retardance(S_m_hat)

    # Orientation ambiguity: theta or theta - pi/2
    diff = np.abs(theta_rec - theta_true)
    diff2 = np.abs(theta_rec - (theta_true - np.pi / 2))
    assert diff < 0.15 or diff2 < 0.15


def test_resolve_fringe_orders():
    # Test multi-fringe resolution
    sigma_diff = 4e6  # 4 MPa -> should be > 1 fringe for some wavelengths
    theta_true = 0.0

    # Calculate true deltas
    deltas = (2 * np.pi * C_VALUES * NU * L / WAVELENGTHS) * sigma_diff
    # Simulate wrapped/folded measurement
    # We simulate what invert_wrapped_retardance produces: 2*asin(|sin(delta/2)|)
    delta_folded = 2 * np.arcsin(np.abs(np.sin(deltas / 2)))

    rec_sigma = resolve_fringe_orders(
        delta_folded, theta_true, WAVELENGTHS, C_VALUES, NU, L, sigma_max=10e6, n_max=6
    )

    print(f"True: {sigma_diff}, Rec: {rec_sigma}")
    assert np.isclose(rec_sigma, sigma_diff, rtol=0.1)
