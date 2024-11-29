import numpy as np
 
from findpeaks import findpeaks

"""findpeaksz局部最大值搜索
使用pyximorse模块进行信号相似性搜索。首先，我们创建了一个正弦波信号，并初始化了pyximorse模块。
然后，我们设置了搜索窗口和相似度阈值，并调用了搜索函数。最后，我们遍历并打印了搜索结果。
"""

import numpy as np
import pyximorse
 
# 假设已经有一个初始化好的pyximorse模块和一个信号
# 这里我们使用一个简单的正弦波作为示例信号
fs = 1000  # 采样频率
t = np.arange(0, 1, 1/fs)  # 时间向量
f = 5  # 信号频率
signal = np.sin(2 * np.pi * f * t)
 
# 初始化pyximorse模块
pyximorse.init()
 
# 设置搜索参数
search_window = 0.5  # 秒
threshold = 0.8  # 相似度阈值
 
# 执行相似性搜索
results = pyximorse.search_similar(signal, fs, search_window, threshold)
 
# 输出搜索结果
for result in results:
    print(f"相似位置: {result[0]}s, 相似度: {result[1]}")
