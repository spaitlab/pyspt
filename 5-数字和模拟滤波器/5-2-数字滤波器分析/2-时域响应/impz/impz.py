import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def impz(b, a=None, *args):
    if a is None:
        # b is SOS matrix or single vector (transfer function)
        if np.ndim(b) == 2 and np.size(b, 1) == 6:
            isTF = False  # SOS matrix
            sos = b
        else:
            isTF = True  # single transfer function vector
            sos = None  # not used in this case
        
        if len(args) > 0:
            n = int(args[0])
        else:
            n = None
        
        if len(args) > 1:
            Fs = float(args[1])
        else:
            Fs = 1.0
    
    else:
        # b and a are coefficient vectors (transfer function)
        isTF = True
        sos = None
        
        n = args[0] if len(args) > 0 else None
        Fs = args[1] if len(args) > 1 else 1.0
    
    # Compute impulse response
    if n is None:
        # Determine length of impulse response automatically
        if isTF:
            impulse_response = signal.impulse((b, a), N=1000)
        else:
            impulse_response = signal.sosfilt(sos, np.array([1] + [0] * 999))
    else:
        # User specified length of impulse response
        impulse_response = np.zeros(n + 1)
        impulse_response[0] = 1
        if isTF:
            impulse_response = signal.lfilter(b, a, impulse_response)
        else:
            impulse_response = signal.sosfilt(sos, impulse_response)
    
    # Compute time vector
    t = np.arange(len(impulse_response)) / Fs
    
    return impulse_response, t

# # 示例，验证其有效性
# b = np.array([1, 0.5, 0.25])
# a = np.array([1, -0.6, 0.1])
# h, t = impz(b, a, 50, 1.0)

# # Plot impulse response
# plt.stem(t, h)
# plt.xlabel('Time')
# plt.ylabel('Impulse Response')
# plt.title('Impulse Response of a System')
# plt.grid(True)
# plt.show()