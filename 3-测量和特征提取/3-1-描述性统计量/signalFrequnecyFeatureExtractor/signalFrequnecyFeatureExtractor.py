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
