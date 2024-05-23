import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize_scalar


def firpmord(f, a, dev, fs):
    """
    Estimate the minimum order for a Parks-McClellan (firpm) FIR filter.

    Parameters:
        f (array_like): Cutoff frequencies.
        a (array_like): Desired amplitudes.
        dev (array_like): Maximum passband and minimum stopband ripple.
        fs (float): Sampling frequency.

    Returns:
        n (int): Minimum filter order.
        fo (ndarray): Optimal frequencies.
        ao (ndarray): Optimal amplitudes.
        w (ndarray): Optimal weights.
    """
    assert len(f) == len(a) == len(dev) == 2, "Input arrays must have length 2."

    rp, rs = dev
    delta_p = (10 ** (rp / 20) - 1) / (10 ** (rp / 20) + 1)
    delta_s = 10 ** (-rs / 20)

    def objective(n):
        delta = np.array([delta_p, delta_s])
        m = (n - 1) // 2
        k = np.arange(1, m + 1)
        h = np.sinc(f[1] * k / fs) - np.sinc(f[0] * k / fs)
        a_m = 2 * np.sum(np.cos(2 * np.pi * f[1] * k / fs) * h)
        d_m = 2 * np.sum(np.cos(2 * np.pi * f[1] * k / fs) * np.sin(np.pi * k / m) / (np.pi * k))
        d_m += np.sum(np.cos(2 * np.pi * f[0] * k / fs) * np.sin(np.pi * k / m) / (np.pi * k))

        # Modify alpha calculation to ensure shape compatibility
        alpha = np.sum(np.abs(1 + (-1) ** k * a_m))
        delta_a = alpha - 1
        delta_d = (1 - d_m) / 2
        return delta @ np.array([delta_a, delta_d])

    res = minimize_scalar(objective, method='bounded', bounds=(1, 1000))
    n = int(np.ceil(res.x))

    m = (n - 1) // 2
    fo = np.array([0, f[0] / fs, f[1] / fs, 0.5])
    ao = np.array([a[0], a[0], a[1], a[1]])
    w = np.array([1, delta_p / delta_s, 1])

    return n, fo, ao, w