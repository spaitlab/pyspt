# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：findsignal

函数来源：[MATLAB findsignal]


### 说明
     signalFrequencyFeatureExtractor是一个用于信号处理的MATLAB函数，它可以提取信号的频率特征，如功率谱密度、谱峰值等。
% 创建一个简单的正弦波信号
Fs = 1000;           % 采样频率
t = 0:1/Fs:1;         % 时间向量
f = 5;               % 信号频率
signal = sin(2*pi*f*t); % 信号
 
% 使用signalFrequencyFeatureExtractor提取频率特征
[Pxx, f] = signalFrequencyFeatureExtractor(signal, Fs, 'FeatureType', 'POWER SPECTRUM');
 
% 绘制功率谱
plot(f, Pxx);
title('Power Spectrum');
xlabel('Frequency (Hz)');
ylabel('Power');

## Python函数描述：signalFrequencyFeatureExtractor
##函数来源：自定义
##说明
signalFrequencyFeatureExtractor是一个用于提取信号频率特征的Python函数。
这个函数需要一个信号数据集和一个频率列表作为输入，然后计算这些频率下信号的特征。

import numpy as np


 """signalFrequencyFeatureExtractor是一个用于提取信号频率特征的Python函数。
这个函数需要一个信号数据集和一个频率列表作为输入，然后计算这些频率下信号的特征。

"""
def signalFrequencyFeatureExtractor(signal, frequency_list):
    # 假设信号是一个一维数组，频率列表是一个包含多个频率的列表
    # 计算每个频率下信号的特征，这里以均值和标准差作为示例
    features = []
    for frequency in frequency_list:
        # 假设信号采样频率为fs，时长为t
        # 计算每个频率的信号切片长度
        n = int(signal.shape[0] * (1 / frequency))
        # 对信号进行重采样
        resampled_signal = np.reshape(signal, (n, int(signal.shape[0] / n)))
        # 计算均值和标准差
        mean_value = np.mean(resampled_signal, axis=1)
        std_value = np.std(resampled_signal, axis=1)
        # 将特征拼接成一个向量
        feature_vector = np.hstack((mean_value, std_value))
        features.append(feature_vector)
    
    return np.array(features)
 
# 示例信号和频率
signal = np.random.random(1000)  # 随机生成一个信号
frequency_list = [10, 20, 30]  # 需要计算特征的频率列表
 
# 提取特征
features = signalFrequencyFeatureExtractor(signal, frequency_list)
print(features)


