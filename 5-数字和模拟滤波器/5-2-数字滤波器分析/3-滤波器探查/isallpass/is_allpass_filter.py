import numpy as np

def is_allpass_filter(b, a):
    # 计算滤波器的零点和极点
    zeros = np.roots(b)
    poles = np.roots(a)
    
    # 判断是否为全通滤波器
    if len(zeros) != len(poles):
        return False
    
    reciprocal_zeros = 1 / np.conj(zeros)
    reciprocal_zeros = np.sort(reciprocal_zeros)
    poles = np.sort(poles)
    
    return np.allclose(reciprocal_zeros, poles, rtol=1e-5, atol=1e-8)