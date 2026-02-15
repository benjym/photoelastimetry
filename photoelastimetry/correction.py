import numpy as np


def compute_disorder_correction(N, order_param):
    """
    Compute the correction factor for random stress orientation.

    Parameters
    ----------
    N : float
        Number of independent stress states along the path.
    order_param : float
        Order parameter [0, 1] (denoted as |m|).

    Returns
    -------
    K : float
        Correction factor.
    """
    if N <= 0:
        return 1.0

    # Avoid division by zero if N is very small (though N should be >= 1 physically)
    inv_N = 1.0 / N

    correction = 1.0 / np.sqrt(inv_N + order_param**2)
    return correction


def estimate_grain_encounters(nu, L, d):
    """
    Estimate number of grain encounters.

    Parameters
    ----------
    nu : float
        Solid fraction.
    L : float
        Thickness (m).
    d : float
        Particle diameter (m).

    Returns
    -------
    N : float
        Estimated number of encounters.
    """
    if d <= 0:
        return 1.0

    return 1.5 * (nu * L) / d
