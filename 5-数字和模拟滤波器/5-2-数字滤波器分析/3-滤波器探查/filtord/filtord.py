import numpy as np
from scipy.signal import firwin, iirfilter

def filtord(b, *args):
    if len(args) == 0:
        a = [1]  # Assume FIR for now
    else:
        a = args[0]
    a = np.asarray(a)
    
    # Cast to precision rules
    if np.issubdtype(b.dtype, np.float32) or np.issubdtype(a.dtype, np.float32):
        convClass = np.float32
    else:
        convClass = np.float64
    
    b = b.astype(convClass)
    a = a.astype(convClass)

    # If b is SOS or vector
    if len(args) == 0:
        a1 = a
        if np.ndim(b) == 1:
            # If input is column vector, transpose to obtain row vectors
            if b.ndim == 2 and b.shape[1] == 6:
                # Check if input is a valid SOS matrix
                raise ValueError('Invalid SOS matrix dimensions')
            b1 = b.reshape(1, -1)
        else:
            # If input is a matrix, check if it is a valid SOS matrix
            if b.shape[1] != 6:
                raise ValueError('Invalid SOS matrix dimensions')
            b1, a1 = sos2tf(b)
    else:  # If b and a are vectors
        # If b is not a vector, then only one input is supported
        if b.ndim > 1 and b.shape[0] > 1 and b.shape[1] > 1:
            raise ValueError('Invalid number of inputs')
        
        # If a is not a vector
        if a.ndim > 1 and a.shape[0] > 1 and a.shape[1] > 1:
            raise ValueError('Input not supported')
        
        b1 = b
        a1 = a
        
        
        # If input is column vector, transpose to obtain row vectors
        if b.ndim == 2 and b.shape[1] == 1:
            b1 = b.T
        
        if a.ndim == 2 and a.shape[1] == 1:
            a1 = a.T

    # Normalizing the filter coefficients
    if not np.allclose(b1, 0):
        maxCoefNum = np.max(np.abs(b1))
        b1 /= maxCoefNum
    
    if not np.allclose(a1, 0):
        maxCoefDen = np.max(np.abs(a1))
        a1 /= maxCoefDen
    
    # Returning the index of the last nonzero coefficient
    nZeroLastNum = np.flatnonzero(b1)[-1] if not np.allclose(b1, 0) else 0
    nZeroLastDen = np.flatnonzero(a1)[-1] if not np.allclose(a1, 0) else 0
    
    # Filter order n is maximum of the last nonzero coefficient subtracted by 1
    n = max(nZeroLastNum, nZeroLastDen)
    
    return n


# Placeholder for sos2tf function (not implemented in this example)
def sos2tf(b):
    # Implement SOS to TF conversion if needed
    raise NotImplementedError("sos2tf function is not implemented")


# # Example usage
# b = np.array([1, -0.5, 0.2])
# a = np.array([1, -0.3])

# n = filtord(b, a)
# print("Filter order (n):", n)

# # 示例1
# # 设计 FIR 滤波器
# order = 20  # 滤波器阶数
# cutoff_freq = 0.5  # 截止频率

# # 计算截止频率对应的归一化频率
# nyquist_freq = 1  # Nyquist 频率，即采样频率的一半
# normalized_cutoff_freq = cutoff_freq / nyquist_freq

# # 使用 firwin 函数设计 FIR 滤波器
# b = firwin(order + 1, normalized_cutoff_freq, window='hamming')
# n = filtord(b)
# print("Filter order (n):", n)


# # 设计低通 FIR 滤波器（equiripple 方法）
# fir_passband_freq = 100  # 通带频率
# fir_stopband_freq = 120  # 阻带频率
# fir_passband_ripple = 0.5  # 通带波纹
# fir_stopband_attenuation = 60  # 阻带衰减

# # 计算归一化频率
# nyquist_freq = 0.5* 1000  # 采样频率的一半
# normalized_passband_freq = fir_passband_freq / nyquist_freq
# normalized_stopband_freq = fir_stopband_freq / nyquist_freq

# # 计算滤波器系数（FIR）
# fir_coeffs = firwin(numtaps=115, cutoff=normalized_passband_freq, \
#                          width=None, window='hamming', pass_zero=True, \
#                          scale=True, fs=None)
# n = filtord(fir_coeffs)
# print("Filter order (n):", n)

# # 设计低通 IIR 滤波器（Butterworth 方法）
# iir_passband_freq = 100  # 通带频率
# iir_stopband_freq = 120  # 阻带频率
# iir_passband_ripple = 0.5  # 通带波纹
# iir_stopband_attenuation = 60  # 阻带衰减

# # 计算归一化频率
# nyquist_freq = 0.5 * 1000  # 采样频率的一半
# normalized_passband_freq = iir_passband_freq / nyquist_freq
# normalized_stopband_freq = iir_stopband_freq / nyquist_freq

# # 设计滤波器（IIR）
# b, a = iirfilter(41, normalized_passband_freq, btype='low', analog=False, ftype='butter', output='ba', fs=1000)
# n = filtord(b,a)
# print("Filter order (n):", n)