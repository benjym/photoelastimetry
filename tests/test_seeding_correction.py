import numpy as np

from photoelastimetry.image import predict_stokes
from photoelastimetry.seeding import PhaseDecomposedSeed, phase_decomposed_seeding, retardance_to_delta_sigma


def test_phase_decomposed_seeding_with_correction():
    # Setup dummy data [H, W, n_wl, n_angles]
    H, W = 10, 10
    n_wl = 3
    wavelengths = np.array([650e-9, 550e-9, 450e-9])
    C_values = np.array([2e-12, 2e-12, 2e-12])
    nu = 1.0
    L = 0.01  # 10 mm
    S_i_hat = np.array([1.0, 0.0, 0.0])

    # Create synthetic data with non-zero stress
    sigma_diff = 5e6  # 5 MPa
    theta_true = np.pi / 6

    s_xx = sigma_diff / 2 * (1 + np.cos(2 * theta_true))
    s_yy = sigma_diff / 2 * (1 - np.cos(2 * theta_true))
    s_xy = sigma_diff / 2 * np.sin(2 * theta_true)

    # Generate Stokes vectors
    S_m_hat = np.stack(
        [predict_stokes(s_xx, s_yy, s_xy, C_values[i], nu, L, wavelengths[i], S_i_hat) for i in range(n_wl)],
        axis=0,
    )  # [n_wl, 2]

    # Unpack to intensities (simplified inverse of compute_stokes)
    # S1 = I0 - I90, S2 = I45 - I135, S0 = I0 + I90 = I45 + I135
    # Let S0 = 1 (normalized)
    # I0 = (1 + S1)/2, I90 = (1 - S1)/2
    # I45 = (1 + S2)/2, I135 = (1 - S2)/2

    I_0 = (1 + S_m_hat[:, 0]) / 2
    I_90 = (1 - S_m_hat[:, 0]) / 2
    I_45 = (1 + S_m_hat[:, 1]) / 2
    I_135 = (1 - S_m_hat[:, 1]) / 2

    # Broadcast to image
    data = np.zeros((H, W, n_wl, 4))
    for i in range(n_wl):
        data[:, :, i, 0] = I_0[i]
        data[:, :, i, 1] = I_45[i]
        data[:, :, i, 2] = I_90[i]
        data[:, :, i, 3] = I_135[i]

    # Run WITHOUT correction (baseline)
    seed_base = phase_decomposed_seeding(
        data, wavelengths, C_values, nu, L, correction_params={"enabled": False}
    )

    # Run WITH correction (N=100, |m|=0.5)
    # Correction factor should be > 1
    # K = 1/sqrt(0.01 + 0.25) = 1/sqrt(0.26) approx 1.96

    seed_corrected = phase_decomposed_seeding(
        data,
        wavelengths,
        C_values,
        nu,
        L,
        correction_params={"enabled": True, "order_param": 0.5, "N": 100.0},
    )

    assert isinstance(seed_base, PhaseDecomposedSeed)
    assert isinstance(seed_corrected, PhaseDecomposedSeed)

    diff_base = seed_base.delta_sigma
    diff_corr = seed_corrected.delta_sigma

    # Filter valid pixels (should be all of them)
    mask = diff_base > 1e4
    assert np.sum(mask) > 0

    ratio = diff_corr[mask] / diff_base[mask]
    mean_ratio = np.mean(ratio)

    expected_factor = 1.0 / np.sqrt(0.01 + 0.25)

    print(f"Mean ratio: {mean_ratio}, Expected: {expected_factor}")
    assert np.isclose(mean_ratio, expected_factor, rtol=0.01)
    assert np.allclose(
        retardance_to_delta_sigma(seed_base.retardance, wavelengths, C_values, nu, L), seed_base.delta_sigma
    )
    assert np.allclose(
        retardance_to_delta_sigma(seed_corrected.retardance, wavelengths, C_values, nu, L),
        seed_corrected.delta_sigma,
    )


def test_phase_decomposed_seeding_with_diameter():
    # Test passing 'd' instead of 'N'
    H, W = 2, 2
    data = np.ones((H, W, 3, 4))
    wavelengths = np.array([650e-9, 550e-9, 450e-9])
    C_values = np.array([2e-12, 2e-12, 2e-12])

    seed_d = phase_decomposed_seeding(
        data,
        wavelengths,
        C_values,
        0.6,
        0.01,
        correction_params={"enabled": True, "order_param": 0.5, "d": 0.001},  # 1 mm -> N approx 9
    )

    assert isinstance(seed_d, PhaseDecomposedSeed)
    assert seed_d.retardance.shape == (H, W, 3)
    assert seed_d.theta.shape == (H, W)
    assert seed_d.delta_sigma.shape == (H, W)
    assert seed_d.to_stress_map(K=0.5).shape == (H, W, 3)
