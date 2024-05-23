import numpy as np


def square(t, duty=50):
    """
    Generate a square wave with a period of 2*pi based on a given time vector t.

    Parameters:
    t : array_like
        Time vector.
    duty : float, optional
        Duty cycle percentage. Default is 50%.

    Returns:
    s : ndarray
        An array representing the square wave signal.
    """
    tmp = np.mod(t, 2 * np.pi)  # Normalize t to the range [0, 2*pi)
    w0 = (2 * np.pi * duty) / 100  # Compute the threshold for the duty cycle
    s = np.where(tmp < w0, 1, -1)  # Create the square wave signal using np.where
    return s