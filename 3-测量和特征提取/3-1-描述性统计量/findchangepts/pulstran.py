import numpy as np

def pulstran(t, d, x):
    y = np.zeros_like(t)
    for i in range(len(d)):
        y += d[i, 1] * x(t - d[i, 0])
    return y

# 定义矩形脉冲函数
def rectpuls(t):
    return np.where(np.abs(t) <= 0.5, 1, 0)