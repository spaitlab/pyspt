import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import freqz


def firpm(N, f, a):
    """
    Python implementation of firpm function similar to MATLAB's firpm.

    Parameters:
        N (int): Filter order.
        f (array_like): Array of frequency points where the amplitude values are specified.
                        The frequencies are normalized to the Nyquist frequency, so they must be between 0 and 1.
        a (array_like): Array of amplitude values corresponding to the frequency points in f.

    Returns:
        b (ndarray): Coefficients of the filter.
    """
    # Ensure that f and a have the same length
    if len(f) != len(a):
        raise ValueError("Lengths of frequency and amplitude arrays must be the same.")

    # Design the filter using Parks-McClellan algorithm
    b = signal.firwin2(N, f, a)

    return b