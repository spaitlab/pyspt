import numpy as np
import cupy as cp

def fftfilt(b, x):
    """
    使用 FFT 在 GPU 上应用 FIR 滤波器。
    
    参数:
    b : array_like
        FIR 滤波器的系数（一维数组）。
    x : array_like
        要滤波的信号（一维数组）。
    
    返回:
    y : ndarray
        滤波后的信号数组。
    """
    # 将数据和滤波器系数传输到 GPU
    x_gpu = cp.asarray(x)
    b_gpu = cp.asarray(b)

    # 使用 FFT 进行滤波 (频域卷积)
    y_gpu = cp.fft.ifft(cp.fft.fft(x_gpu) * cp.fft.fft(b_gpu, n=len(x_gpu)))
    
    # 将结果转换回 CPU 并取实部
    y = cp.asnumpy(y_gpu).real
    
    return y
