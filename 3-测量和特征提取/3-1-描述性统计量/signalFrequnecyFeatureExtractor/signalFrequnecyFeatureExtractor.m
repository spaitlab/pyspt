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