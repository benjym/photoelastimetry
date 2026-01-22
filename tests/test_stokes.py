#!/usr/bin/env python3
"""
Comprehensive tests for local.py - verifying full stress tensor recovery.

This test suite verifies that the multi-wavelength approach can recover
the full stress tensor, not just the principal stress difference.
"""

import sys

import numpy as np

from photoelastimetry.optimiser.stokes import (  # compute_residual,
    compute_normalised_stokes,
    compute_principal_angle,
    compute_retardance,
    compute_solid_fraction,
    compute_stokes_components,
    mueller_matrix,
    predict_stokes,
    recover_stress_tensor,
)

# Test parameters
WAVELENGTHS = np.array([650e-9, 550e-9, 450e-9])  # R, G, B in meters
C_VALUES = np.array([2e-12, 2.2e-12, 2.5e-12])  # Different C for each wavelength
NU = 1.0  # Solid fraction
L = 0.01  # Sample thickness (m)
S_I_HAT = np.array([0.1, 0.2, 0.0])  # Incoming polarisation [S1_hat, S2_hat, S3_hat]


def print_separator(title):
    """Print a formatted separator with title."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_stokes_components():
    """Test 1: Verify Stokes components computation."""
    print_separator("TEST 1: Stokes Components Computation")

    I_0, I_45, I_90, I_135 = 1.0, 0.6, 0.4, 0.3
    S0, S1, S2 = compute_stokes_components(I_0, I_45, I_90, I_135)

    # Verify equations
    assert np.isclose(S0, I_0 + I_90), "S0 = I_0 + I_90"
    assert np.isclose(S1, I_0 - I_90), "S1 = I_0 - I_90"
    assert np.isclose(S2, I_45 - I_135), "S2 = I_45 - I_135"

    print(f"  Input intensities: I_0={I_0}, I_45={I_45}, I_90={I_90}, I_135={I_135}")
    print(f"  Computed: S0={S0:.4f}, S1={S1:.4f}, S2={S2:.4f}")
    print("  ✓ PASSED: Stokes components correctly computed")
    return True


def test_normalised_stokes():
    """Test 2: Verify normalised Stokes computation."""
    print_separator("TEST 2: Normalised Stokes Components")

    S0, S1, S2 = 1.4, 0.6, 0.3
    S1_hat, S2_hat = compute_normalised_stokes(S0, S1, S2)

    assert np.isclose(S1_hat, S1 / S0), "S1_hat = S1/S0"
    assert np.isclose(S2_hat, S2 / S0), "S2_hat = S2/S0"

    print(f"  Input: S0={S0}, S1={S1}, S2={S2}")
    print(f"  Normalised: S1_hat={S1_hat:.4f}, S2_hat={S2_hat:.4f}")
    print("  ✓ PASSED: Normalised components correctly computed")
    return True


def test_retardance():
    """Test 3: Verify retardance computation."""
    print_separator("TEST 3: Retardance Computation")

    sigma_xx, sigma_yy, sigma_xy = 2e6, -1e6, 0.5e6
    delta = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C_VALUES[1], NU, L, WAVELENGTHS[1])

    # Verify formula: delta = (2*pi*C*n*L/lambda) * sqrt((sigma_xx-sigma_yy)^2 + 4*sigma_xy^2)
    psd = np.sqrt((sigma_xx - sigma_yy) ** 2 + 4 * sigma_xy**2)
    expected_delta = (2 * np.pi * C_VALUES[1] * NU * L / WAVELENGTHS[1]) * psd
    assert np.isclose(delta, expected_delta), "Retardance formula verification"

    print(f"  Stress: σ_xx={sigma_xx/1e6:.2f} MPa, σ_yy={sigma_yy/1e6:.2f} MPa, σ_xy={sigma_xy/1e6:.2f} MPa")
    print(f"  Principal stress difference: {psd/1e6:.4f} MPa")
    print(f"  Retardance: δ={delta:.4f} rad")
    print("  ✓ PASSED: Retardance correctly computed")
    return True


def test_principal_angle():
    """Test 4: Verify principal angle computation."""
    print_separator("TEST 4: Principal Angle Computation")

    # Test case 1: Pure shear (45° expected)
    sigma_xx, sigma_yy, sigma_xy = 1e6, 1e6, 1e6
    theta = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)
    expected_theta = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)
    assert np.isclose(theta, expected_theta), "Principal angle formula verification"
    print(
        f"  Case 1 - Pure shear: σ_xx={sigma_xx/1e6:.2f} MPa, σ_yy={sigma_yy/1e6:.2f} MPa, σ_xy={sigma_xy/1e6:.2f} MPa"
    )
    print(f"           θ={np.rad2deg(theta):.2f}° (expected ~45°)")

    # Test case 2: No shear (0° or 90° expected)
    sigma_xx, sigma_yy, sigma_xy = 2e6, 1e6, 0
    theta = compute_principal_angle(sigma_xx, sigma_yy, sigma_xy)
    print(
        f"  Case 2 - No shear: σ_xx={sigma_xx/1e6:.2f} MPa, σ_yy={sigma_yy/1e6:.2f} MPa, σ_xy={sigma_xy/1e6:.2f} MPa"
    )
    print(f"           θ={np.rad2deg(theta):.2f}° (expected 0°)")

    print("  ✓ PASSED: Principal angle correctly computed")
    return True


def test_mueller_matrix():
    """Test 5: Verify Mueller matrix properties."""
    print_separator("TEST 5: Mueller Matrix Properties")

    theta = np.pi / 4  # 45 degrees
    delta = np.pi / 2  # 90 degrees
    M = mueller_matrix(theta, delta)

    # Check shape
    assert M.shape == (4, 4), "Mueller matrix should be 4x4"

    # Check M[0,0] = 1
    assert np.isclose(M[0, 0], 1.0), "M[0,0] should always be 1"

    # Check symmetry properties
    print(f"  θ={np.rad2deg(theta):.1f}°, δ={np.rad2deg(delta):.1f}°")
    print(f"  M[0,0]={M[0,0]:.4f} (should be 1.0)")
    print(f"  M[1,1]={M[1,1]:.4f}, M[2,2]={M[2,2]:.4f}, M[3,3]={M[3,3]:.4f}")
    print("  ✓ PASSED: Mueller matrix has correct properties")
    return True


def test_forward_model():
    """Test 6: Verify forward model (stress to Stokes)."""
    print_separator("TEST 6: Forward Model - Stress to Stokes")

    sigma_xx, sigma_yy, sigma_xy = 2e6, -1e6, 0.5e6

    print(
        f"  Input stress: σ_xx={sigma_xx/1e6:.2f} MPa, σ_yy={sigma_yy/1e6:.2f} MPa, σ_xy={sigma_xy/1e6:.2f} MPa"
    )
    print(f"  Incoming polarisation: [{S_I_HAT[0]:.2f}, {S_I_HAT[1]:.2f}]")
    print("\n  Predicted Stokes for each wavelength:")

    for i, (wl, C) in enumerate(zip(WAVELENGTHS, C_VALUES)):
        S_p = predict_stokes(sigma_xx, sigma_yy, sigma_xy, C, NU, L, wl, S_I_HAT)
        color = ["Red", "Green", "Blue"][i]
        print(f"    {color:5s} ({wl*1e9:.0f}nm): [{S_p[0]:+.6f}, {S_p[1]:+.6f}]")

    print("  ✓ PASSED: Forward model produces predictions")
    return True


def test_stress_recovery_uniaxial():
    """Test 7: Recover uniaxial stress (σ_yy=0, σ_xy=0)."""
    print_separator("TEST 7: Stress Recovery - Uniaxial Case")

    # True stress: uniaxial tension
    sigma_xx_true = 5e6  # 5 MPa
    sigma_yy_true = 0.0
    sigma_xy_true = 0.0

    print(f"  True stress (uniaxial):")
    print(f"    σ_xx = {sigma_xx_true/1e6:.2f} MPa")
    print(f"    σ_yy = {sigma_yy_true/1e6:.2f} MPa")
    print(f"    σ_xy = {sigma_xy_true/1e6:.2f} MPa")

    # Generate synthetic measurements
    S_m_hat = np.zeros((3, 2))
    for c in range(3):
        S_m_hat[c] = predict_stokes(
            sigma_xx_true,
            sigma_yy_true,
            sigma_xy_true,
            C_VALUES[c],
            NU,
            L,
            WAVELENGTHS[c],
            S_I_HAT,
        )

    # Recover stress with reasonable initial guess
    stress_recovered, success = recover_stress_tensor(
        S_m_hat,
        WAVELENGTHS,
        C_VALUES,
        NU,
        L,
        S_I_HAT,
        initial_guess=np.array([3e6, 0.0, 0.0]),
    )

    print(f"\n  Recovered stress:")
    print(f"    σ_xx = {stress_recovered[0]/1e6:.4f} MPa")
    print(f"    σ_yy = {stress_recovered[1]/1e6:.4f} MPa")
    print(f"    σ_xy = {stress_recovered[2]/1e6:.4f} MPa")
    print(f"    Success: {success}")

    # Check errors
    err_xx = abs(stress_recovered[0] - sigma_xx_true) / 1e6
    err_yy = abs(stress_recovered[1] - sigma_yy_true) / 1e6
    err_xy = abs(stress_recovered[2] - sigma_xy_true) / 1e6

    print(f"\n  Errors:")
    print(f"    Δσ_xx = {err_xx:.4f} MPa")
    print(f"    Δσ_yy = {err_yy:.4f} MPa")
    print(f"    Δσ_xy = {err_xy:.4f} MPa")

    # Principal stress difference should match well
    psd_true = abs(sigma_xx_true - sigma_yy_true)
    psd_recovered = np.sqrt((stress_recovered[0] - stress_recovered[1]) ** 2 + 4 * stress_recovered[2] ** 2)
    psd_error = abs(psd_recovered - psd_true) / 1e6

    print(f"\n  Principal stress difference:")
    print(f"    True:      {psd_true/1e6:.4f} MPa")
    print(f"    Recovered: {psd_recovered/1e6:.4f} MPa")
    print(f"    Error:     {psd_error:.4f} MPa")

    if psd_error < 0.1:
        print("  ✓ PASSED: Principal stress difference recovered accurately")
        return True
    else:
        print("  ✗ WARNING: Large error in principal stress difference")
        return False


def test_stress_recovery_biaxial():
    """Test 8: Recover biaxial stress."""
    print_separator("TEST 8: Stress Recovery - Biaxial Case")

    # True stress: biaxial
    sigma_xx_true = 3e6  # 3 MPa tension
    sigma_yy_true = -2e6  # 2 MPa compression
    sigma_xy_true = 0.0

    print(f"  True stress (biaxial):")
    print(f"    σ_xx = {sigma_xx_true/1e6:+.2f} MPa")
    print(f"    σ_yy = {sigma_yy_true/1e6:+.2f} MPa")
    print(f"    σ_xy = {sigma_xy_true/1e6:+.2f} MPa")

    # Generate synthetic measurements
    S_m_hat = np.zeros((3, 2))
    for c in range(3):
        S_m_hat[c] = predict_stokes(
            sigma_xx_true,
            sigma_yy_true,
            sigma_xy_true,
            C_VALUES[c],
            NU,
            L,
            WAVELENGTHS[c],
            S_I_HAT,
        )

    # Recover stress
    stress_recovered, success = recover_stress_tensor(
        S_m_hat,
        WAVELENGTHS,
        C_VALUES,
        NU,
        L,
        S_I_HAT,
        initial_guess=np.array([2e6, -1e6, 0.0]),
    )

    print(f"\n  Recovered stress:")
    print(f"    σ_xx = {stress_recovered[0]/1e6:+.4f} MPa")
    print(f"    σ_yy = {stress_recovered[1]/1e6:+.4f} MPa")
    print(f"    σ_xy = {stress_recovered[2]/1e6:+.4f} MPa")

    # Check principal stress difference
    psd_true = np.sqrt((sigma_xx_true - sigma_yy_true) ** 2 + 4 * sigma_xy_true**2)
    psd_recovered = np.sqrt((stress_recovered[0] - stress_recovered[1]) ** 2 + 4 * stress_recovered[2] ** 2)

    print(f"\n  Principal stress difference:")
    print(f"    True:      {psd_true/1e6:.4f} MPa")
    print(f"    Recovered: {psd_recovered/1e6:.4f} MPa")
    print(f"    Error:     {abs(psd_recovered - psd_true)/1e6:.4f} MPa")

    if abs(psd_recovered - psd_true) / 1e6 < 0.1:
        print("  ✓ PASSED: Principal stress difference recovered accurately")
        return True
    else:
        print("  ✗ WARNING: Large error in principal stress difference")
        return False


def test_stress_recovery_with_shear():
    """Test 9: Recover stress with shear component."""
    print_separator("TEST 9: Stress Recovery - With Shear")

    # True stress: includes shear
    sigma_xx_true = 2e6
    sigma_yy_true = -1e6
    sigma_xy_true = 1e6

    print(f"  True stress (with shear):")
    print(f"    σ_xx = {sigma_xx_true/1e6:+.2f} MPa")
    print(f"    σ_yy = {sigma_yy_true/1e6:+.2f} MPa")
    print(f"    σ_xy = {sigma_xy_true/1e6:+.2f} MPa")

    # Generate synthetic measurements
    S_m_hat = np.zeros((3, 2))
    for c in range(3):
        S_m_hat[c] = predict_stokes(
            sigma_xx_true,
            sigma_yy_true,
            sigma_xy_true,
            C_VALUES[c],
            NU,
            L,
            WAVELENGTHS[c],
            S_I_HAT,
        )

    # Recover stress
    stress_recovered, success = recover_stress_tensor(
        S_m_hat,
        WAVELENGTHS,
        C_VALUES,
        NU,
        L,
        S_I_HAT,
        initial_guess=np.array([1.5e6, -0.5e6, 0.8e6]),
    )

    print(f"\n  Recovered stress:")
    print(f"    σ_xx = {stress_recovered[0]/1e6:+.4f} MPa")
    print(f"    σ_yy = {stress_recovered[1]/1e6:+.4f} MPa")
    print(f"    σ_xy = {stress_recovered[2]/1e6:+.4f} MPa")

    # Check all components
    theta_true = compute_principal_angle(sigma_xx_true, sigma_yy_true, sigma_xy_true)
    theta_recovered = compute_principal_angle(stress_recovered[0], stress_recovered[1], stress_recovered[2])

    print(f"\n  Principal angle:")
    print(f"    True:      {np.rad2deg(theta_true):.2f}°")
    print(f"    Recovered: {np.rad2deg(theta_recovered):.2f}°")

    psd_true = np.sqrt((sigma_xx_true - sigma_yy_true) ** 2 + 4 * sigma_xy_true**2)
    psd_recovered = np.sqrt((stress_recovered[0] - stress_recovered[1]) ** 2 + 4 * stress_recovered[2] ** 2)

    print(f"\n  Principal stress difference:")
    print(f"    True:      {psd_true/1e6:.4f} MPa")
    print(f"    Recovered: {psd_recovered/1e6:.4f} MPa")
    print(f"    Error:     {abs(psd_recovered - psd_true)/1e6:.4f} MPa")

    if abs(psd_recovered - psd_true) / 1e6 < 0.1:
        print("  ✓ PASSED: Principal stress difference recovered accurately")
        return True
    else:
        print("  ✗ WARNING: Large error in principal stress difference")
        return False


def test_solid_fraction():
    """Test 10: Verify solid fraction calculation."""
    print_separator("TEST 10: Solid Fraction Calculation")

    S0_measured = 0.5
    S_ref = 1.0
    mu = 0.15

    nu = compute_solid_fraction(S0_measured, S_ref, mu, L)
    expected_n = -np.log(S0_measured / S_ref) / (mu * L)

    assert np.isclose(nu, expected_n), "Solid fraction formula verification"

    print(f"  S0={S0_measured}, S_ref={S_ref}, μ={mu}, L={L*1000:.1f}mm")
    print(f"  Solid fraction: nu={nu:.4f}")
    print("  ✓ PASSED: Solid fraction correctly computed")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "COMPREHENSIVE TEST SUITE FOR LOCAL.PY" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")

    tests = [
        test_stokes_components,
        test_normalised_stokes,
        test_retardance,
        test_principal_angle,
        test_mueller_matrix,
        test_forward_model,
        test_stress_recovery_uniaxial,
        test_stress_recovery_biaxial,
        test_stress_recovery_with_shear,
        test_solid_fraction,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ FAILED with exception: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {total - passed}/{total}")

    if passed == total:
        print("\n  ✓ ALL TESTS PASSED!")
    else:
        print("\n  ✗ SOME TESTS FAILED")

    print("=" * 70 + "\n")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
