# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：findsignal

函数来源：[MATLAB findsignal]

### 语法

[istart,istop,dist] = findsignal(data,signal)
[istart,istop,dist] = findsignal(data,signal,Name,Value)
findsignal(___)

### 说明
        [istart，istop，dist] = findsignal(data，signal)返回与搜索数组信号最匹配的数据数组数据段的开始和停止索引。最佳匹配的片段是这样的，dist，片段和搜索阵列之间的平方欧几里得距离，是最小的。如果数据和信号是矩阵，那么findsignal会查找与信号最匹配的数据区域的开始列和结束列。在这种情况下，数据和信号必须具有相同的行数。

        [istart，istop，dist] = findsignal(data，signal，Name，Value)使用名称-值对参数指定附加选项。选项包括要应用的归一化、要报告的线段数以及要使用的距离度量。

不带输出参数的findsignal(___)绘制数据并突出显示任何已识别的信号实例。

如果数组是实向量，该函数将数据显示为样本数的函数。

如果数组是复数向量，该函数将在Argand图上显示数据。

如果数组是实矩阵，该函数使用imagesc在一个子绘图上显示信号，在另一个子绘图上显示带有突出显示区域的数据。

如果数组是复矩阵，该函数在每个图像的上半部分和下半部分绘制它们的实部和虚部。

ts = 0:1/fs:0.15;
signal = cos(2*pi*10*ts);
 
subplot(2,1,1)
plot(t,data)
title('Data')
subplot(2,1,2)
plot(ts,signal)
title('Signal')

## Python函数描述：findsignal


使用pyximorse模块进行信号相似性搜索。首先，我们创建了一个正弦波信号，并初始化了pyximorse模块。然后，我们设置了搜索窗口和相似度阈值，并调用了搜索函数。最后，我们遍历并打印了搜索结果。

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


