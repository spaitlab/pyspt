import numpy as np
import matplotlib.pyplot as plt
def gaussdesign(bt, span=3, sps=2):
    """
    Design a Gaussian FIR Pulse-Shaping Filter.

    Parameters:
        bt (float): 3 dB bandwidth-symbol time product.
        span (int, optional): Total length of the filter in symbols. Default is 3.
        sps (int, optional): Number of samples per symbol. Default is 2.

    Returns:
        h (ndarray): Coefficients of the Gaussian filter.
    """
    # Check if filter order is even
    sps_span = sps * span
    if sps_span % 2 != 0:
        raise ValueError("Filter order must be even.")

    # Calculate filter length
    filt_len = sps_span + 1

    # Convert to t in which to compute the filter coefficients
    t = np.linspace(-span / 2, span / 2, filt_len)

    # Compute alpha
    alpha = np.sqrt(np.log(2) / 2) / bt

    # Compute filter coefficients
    h = (np.sqrt(np.pi) / alpha) * np.exp(-(t * np.pi / alpha) ** 2)

    # Normalize coefficients
    h /= np.sum(h)

    return h