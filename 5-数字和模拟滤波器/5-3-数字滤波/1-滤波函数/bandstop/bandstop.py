import numpy as np
from scipy.signal import butter, lfilter

def bandstop(data, freq_band, fs, order=5):
    """
    Apply a bandstop filter to the data.
    
    Parameters:
    data : array_like
        Input signal array.
    freq_band : array_like
        List or array of two elements defining the stop band frequencies, [low cut, high cut] in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the filter, default is 5.
    
    Returns:
    y : ndarray
        Filtered signal array.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    low = freq_band[0] / nyq
    high = freq_band[1] / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y
