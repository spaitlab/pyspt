"""
tf2zp函数用于将传递函数转换为零极点形式，即将传递函数的分子分母多项式系数输入，返回该传递函数的零和极点，python中可以使用可以使用scipy库实现
"""
import numpy as np
from scipy import signal


def tf2zp(num, den):
    # 调用scipy中的tf2zpk函数
    z, p, k = signal.tf2zpk(num, den)

    # 将复数根按照实部和虚部分开
    zeros_real = np.real(z)
    zeros_imag = np.imag(z)
    poles_real = np.real(p)
    poles_imag = np.imag(p)

    return zeros_real, zeros_imag, poles_real, poles_imag, k
