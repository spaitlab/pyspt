import control as ctl
import numpy as np


def isstable(b, a):
    system = ctl.TransferFunction(b, a)

    # 获取系统的极点
    poles = ctl.poles(system)

    # 判断系统是否稳定（所有极点的实部需要小于0）
    return np.all(np.real(poles) < 0)

# 定义传递函数，例如：H(s) = 1 / (s^2 + s + 1)
b = [1]  # 分子
a = [1, 1, -10]  # 分母
print("系统是否稳定？", isstable(b, a))
