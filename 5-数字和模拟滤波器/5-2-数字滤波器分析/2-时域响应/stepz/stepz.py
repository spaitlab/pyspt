import numpy as np
from scipy.signal import lfilter, sosfilt, butter, ellip
from impzlength import impzlength
import matplotlib.pyplot as plt

def stepz(b, *args):
    # Determine if b represents a transfer function or an SOS matrix
    if b.ndim == 1 or b.shape[1] == 1:
        isTF = True
    else:
        isTF = False

    # Process optional arguments
    if len(args) > 0:
        a = args[0]
    else:
        a = np.ones_like(b)

    if len(args) > 1:
        n = int(args[1])
    else:
        n =  None

    if len(args) > 2:
        Fs = float(args[2])
    else:
        Fs = 1.0

    # Compute time vector
    if n is None:
        # Determine the length if not specified
        if isTF:
            N = impzlength(b, a)
        else:
            N = impzlength(b)
        M=0
    elif isinstance(n, (list, np.ndarray)) and len(n) > 1:
        # Vector of indices
        NN = np.round(n).astype(int)
        N = max(NN) + 1
        M = min(NN)
    else:
        # Single value of N
        N = int(n)
        M = 0

    tf = np.arange(M, N) / Fs

    # Form input vector
    x = np.ones_like(tf)

    if isTF:
        sf = lfilter(b, a, x)
    else:
        sf = sosfilt(b, x)

    if isinstance(n, (list, np.ndarray)) and len(n) > 1:
        s = sf[NN - M]
        t = tf[NN - M]
    else:
        s = sf
        t = tf
    # Plotting example using matplotlib
    plt.stem(t, s)
    plt.xlabel('n(Samples)')
    plt.ylabel('Amplitude')
    plt.title('Step Response')
    plt.grid(True)
    plt.show()
    return s, t


# Example usage
# Assuming impzlength is implemented and available
# b and a are the filter coefficients or matrices
# b = np.array([1, 0.5, 0.2])
# a = np.array([1, -0.3])
# N = 50
# Fs = 100
# s, t = stepz(b, a, N, Fs)