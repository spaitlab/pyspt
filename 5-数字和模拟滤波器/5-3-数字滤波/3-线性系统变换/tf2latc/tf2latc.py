import numpy as np
from scipy.signal import lfilter_zi


def tf2latc(num, den=None, phase='none'):
    """
    将传递函数转换为格式和阶梯式滤波器的系数。

    参数:
    num (array_like): 传递函数的分子系数。
    den (array_like, 可选): 传递函数的分母系数。如果未提供，默认为FIR滤波器。
    phase (str, 可选): 指定FIR滤波器的相位类型。可选 'min' 最小相位, 'max' 最大相位, 'none' 默认。

    返回:
    tuple: 返回包含格系数和阶梯系数的元组。
    """
    num = np.atleast_1d(num)
    if den is None:
        den = [1]  # 默认为FIR滤波器
    else:
        den = np.atleast_1d(den)

    # 确保分母首项为1
    if den[0] != 1:
        num = num / den[0]
        den = den / den[0]

    # FIR滤波器
    if np.array_equal(den, [1]):
        k = np.poly(num)  # 直接计算多项式的反射系数
        v = []
        if phase == 'max':
            k = np.flip(k)  # 最大相位
        return k, v

    # IIR滤波器
    else:
        # 计算多项式的反射系数
        k = np.poly(den)
        v = lfilter_zi(num, den)  # 计算阶梯系数
        return k, v


# 示例使用
a = [1, 13 / 24, 5 / 8, 1 / 3]
k, v = tf2latc(1, a)  # 输入分子为1，分母为a
print("格系数 k:", k)
print("阶梯系数 v:", v)