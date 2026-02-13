import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from photoelastimetry import plotting


@pytest.mark.smoke
def test_plot_fringe_pattern_smoke(tmp_path):
    intensity = np.linspace(0, 1, 100).reshape(10, 10)
    isoclinic = np.linspace(-np.pi / 2, np.pi / 2, 100).reshape(10, 10)
    out = tmp_path / "fringe.png"

    plotting.plot_fringe_pattern(intensity, isoclinic, filename=str(out))

    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.smoke
def test_show_all_channels_smoke(tmp_path):
    data = np.random.default_rng(0).integers(0, 255, size=(8, 8, 4, 4), dtype=np.uint16)
    out = tmp_path / "channels.png"

    plotting.show_all_channels(data, {"dtype": "uint16"}, filename=str(out))

    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.smoke
def test_plot_optimization_history_smoke(tmp_path):
    stress_params = np.array([[1.0, 2.0, 0.5], [1.1, 2.1, 0.45], [1.2, 2.2, 0.4]])
    residuals = np.array([1e-1, 1e-2, 1e-3])
    s_pred = np.array(
        [
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            [[0.11, 0.19], [0.21, 0.29], [0.31, 0.39]],
            [[0.12, 0.18], [0.22, 0.28], [0.32, 0.38]],
        ]
    )

    history = {
        "all_paths": [
            {
                "stress_params": stress_params,
                "residuals": residuals,
                "S_predicted": s_pred,
                "is_best": True,
            }
        ],
        "best_path_index": 0,
    }
    s_m_hat = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
    out = tmp_path / "history.png"

    plotting.plot_optimization_history(history, s_m_hat, filename=str(out))

    assert out.exists()
    assert out.stat().st_size > 0
