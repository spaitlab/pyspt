import numpy as np
from scipy.signal import correlate

def find_delay(signal1, signal2):
    # 计算信号间的互相关
    correlation = correlate(signal1, signal2, mode='full', method='auto')
    
    # 找到互相关的最大值索引
    lag_index = np.argmax(correlation)
    
    # 计算延迟
    # 因为使用了'full'模式，所以需要调整索引
    delay = -(lag_index - len(signal2) + 1)
    return delay