import numpy as np
def rectpuls(t, width):
    """
    生成矩形脉冲信号

    参数：
    t : array_like
        时间向量
    width : float
        脉冲宽度

    返回：
    x : ndarray
        生成的矩形脉冲信号
    """
    x = np.zeros_like(t)
    x[np.abs(t) <= width / 2] = 1
    return x