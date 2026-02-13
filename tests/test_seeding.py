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
S_I_HAT = np.array([1.0, 0.0, 0.0])  # Linearly polarised light at 0 degrees


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
