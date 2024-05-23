import numpy as np
from scipy.signal import butter, lfilter

def highpass(data, cutoff, fs, order=5):
    """
    对给定数据应用高通滤波器。
    
    参数:
    data : array_like
        输入信号数组。
    cutoff : float
        高通滤波器的截止频率 (单位：Hz)。
    fs : float
        采样频率 (单位：Hz)。
    order : int, optional
        滤波器的阶数，默认为5。
    
    返回:
    y : ndarray
        滤波后的信号数组。
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y
