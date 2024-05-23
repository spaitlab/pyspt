# designfilt.py
import numpy as np
from scipy.signal import firwin, freqz
import matplotlib.pyplot as plt

def design_bandpass_fir(order, cutoff_low, cutoff_high, fs):
    """
    Designs a bandpass FIR filter using the window method.
    
    Parameters:
    order (int): The order of the filter.
    cutoff_low (float): The lower cutoff frequency (in Hz).
    cutoff_high (float): The higher cutoff frequency (in Hz).
    fs (int): The sampling frequency (in Hz).
    
    Returns:
    ndarray: The coefficients of the FIR filter.
    """
    nyq = fs / 2.0
    low = cutoff_low / nyq
    high = cutoff_high / nyq
    coeffs = firwin(order + 1, [low, high], pass_zero=False)
    return coeffs

def visualize_filter_response(coeffs, fs):
    """
    Visualizes the frequency and phase response of the FIR filter.
    
    Parameters:
    coeffs (ndarray): The coefficients of the FIR filter.
    fs (int): The sampling frequency (in Hz).
    """
    w, h = freqz(coeffs, worN=8000)
    w = w * fs / (2 * np.pi)  # Convert from rad/sample to Hz

    plt.figure(figsize=(12, 6))

    # Magnitude response
    plt.subplot(2, 1, 1)
    plt.plot(w, 20 * np.log10(np.abs(h)), 'b')
    plt.title('Magnitude Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)

    # Phase response
    plt.subplot(2, 1, 2)
    plt.plot(w, np.unwrap(np.angle(h)), 'b')
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
