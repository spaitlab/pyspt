# 信号处理仿真与应用 - 数字滤波

## MATLAB函数描述：filtfilt
 

函数来源：[MATLAB filtfilt](https://ww2.mathworks.cn/help/signal/ref/filtfilt.html)

### 语法

y = filtfilt(b,a,x)
y = filtfilt(sos,g,x)
y = filtfilt(d,x)

### 说明

y = filtfilt(b,a,x) 通过正向和反向处理输入数据 x 来执行零相位数字滤波。在对数据进行正向滤波后，该函数会匹配初始条件以最大限度地减少启动和结束时的瞬变，反转滤波后的序列，并让反转后的序列再次通过滤波器。结果具有以下特征：

 - 零相位失真

 - 滤波器传递函数等于原始滤波器传递函数的平方幅值

 -  滤波器阶数是由 b 和 a 指定的滤波器阶数的两倍

不要将 filtfilt 与微分器和希尔伯特 FIR 滤波器结合使用，因为这些滤波器的运算在很大程度上取决于它们的相位响应。
y = filtfilt(sos,g,x) 使用由矩阵 sos 和定标值 g 表示的二阶节 (biquad) 滤波器对输入数据 x 进行零相位滤波。
y = filtfilt(d,x) 使用数字滤波器 d 对输入数据 x 进行零相位滤波。使用 designfilt 根据频率响应设定生成 d。

### 输入参数

b, a — 传递函数系数
向量
传递函数系数，指定为向量。如果使用全极点滤波器，请为 b 输入 1。如果使用全零 (FIR) 滤波器，请为 a 输入 1。
示例: b = [1 3 3 1]/6 和 a = [3 0 1 0]/3 用于指定一个归一化 3 dB 频率为 0.5π 弧度/采样点的三阶巴特沃斯滤波器。
数据类型: single | double

x — 输入信号
向量 | 矩阵 | N 维数组
输入信号，指定为实数值或复数值向量、矩阵或 N 维数组。x 包含的值必须为有限值。x 的长度必须大于滤波器阶数的三倍，定义为 max(length(B)-1, length(A)-1)。除非 x 是行向量，否则该函数沿 x 的第一个数组维度进行运算。如果 x 是行向量，则该函数沿第二个维度进行运算。
示例: cos(pi/4*(0:159))+randn(1,160) 是单通道行向量信号。
示例: cos(pi./[4;2]*(0:159))'+randn(160,2) 是双通道信号。
数据类型: single | double
复数支持: 是

sos — 二阶节系数
矩阵
二阶节系数，指定为矩阵。sos 是一个 K×6 矩阵，其中节数 K 必须大于或等于 2。如果节数小于 2，则该函数将输入视为分子向量。sos 的每行对应于二阶 (biquad) 滤波器的系数。sos 的第 i 行对应于 [bi(1) bi(2) bi(3) ai(1) ai(2) ai(3)]。
示例: s = [2 4 2 6 0 2;3 3 0 6 0 0] 用于指定一个归一化 3 dB 频率为 0.5π 弧度/采样点的三阶巴特沃斯滤波器。
数据类型: single | double

g — 缩放因子
向量
缩放因子，指定为向量。
数据类型: single | double

d — 数字滤波器
digitalFilter 对象
数字滤波器，指定为 digitalFilter 对象。使用 designfilt 根据频率响应设定生成数字滤波器。
示例: d = designfilt("lowpassiir",FilterOrder=3,HalfPowerFrequency=0.5) 用于指定一个归一化 3 dB 频率为 0.5π 弧度/采样点的三阶巴特沃斯滤波器。

### 输出参量

y — 滤波后的信号
向量 | 矩阵 | N 维数组
滤波后的信号，以向量、矩阵或 N 维数组形式返回。
如果 filtfilt 的输入是单精度值，则该函数返回单精度的输出 y。

## Python函数描述：filtfilt

函数来源：scipy.signal

### 函数定义：

scipy.signal.filtfilt是一个用于零相位滤波（zero-phase filtering）的函数，它可以应用于一维信号。
零相位滤波意味着信号在时域上不会发生相位延迟，也就是说，它会对信号进行前向和反向的滤波，以消除因滤波引入的任何相位延迟。

### 参数
y = filtfilt(b, a, x)
- b和a是滤波器的系数，可以通过scipy.signal.butter、scipy.signal.bessel等函数获得。
- x是输入的一维信号。

### 返回值

- 输出的滤波后的信号。

### 注意事项

- 

### 函数工作原理
scipy.signal.filtfilt函数的工作原理主要基于两个主要步骤：前向滤波和反向滤波。
- 前向滤波：
使用给定的滤波器系数 b 和 a 对输入信号 x 进行正向滤波。
正向滤波的结果是 y1。
- 反向滤波：
对 y1 进行反向滤波，再次使用滤波器系数 b 和 a。
反向滤波的结果是 y2。
- 合并输出：
将 y1 和 y2 的反转（或镜像）版本合并。
由于是零相位滤波，y2 需要反转，然后与 y1 进行拼接。
可以表示为：
y=concatenate(y1,reverse(y2))

### 使用场景

scipy.signal.filtfilt函数由于其零相位延迟的特性，适用于多种信号处理应用场景。在心电图分析中，准确的相位信息对于检测和诊断心脏异常至关重要。filtfilt可以帮助去除噪声，同时保留信号的相位信息。

### 改进建议


## Prompt 1 ： 应用 scipy.signal.filtfilt 函数做MATLAB示例对应

这是MATLAB示例
```
wform = ecg(500);
d1 = designfilt("lowpassiir",FilterOrder=12, ...
    HalfPowerFrequency=0.15,DesignMethod="butter");
y = filtfilt(d1,x);

subplot(1,1,1)
plot(x)
hold on
plot(y,LineWidth=3)
legend("Noisy ECG","Zero-Phase Filtering")

function x = ecg(L)
%ECG Electrocardiogram (ECG) signal generator.
%   ECG(L) generates a piecewise linear ECG signal of length L.
%
%   EXAMPLE:
%   x = ecg(500).';
%   y = sgolayfilt(x,0,3); % Typical values are: d=0 and F=3,5,9, etc. 
%   y5 = sgolayfilt(x,0,5); 
%   y15 = sgolayfilt(x,0,15); 
%   plot(1:length(x),[x y y5 y15]);

%   Copyright 1988-2002 The MathWorks, Inc.

a0 = [0,1,40,1,0,-34,118,-99,0,2,21,2,0,0,0]; % Template
d0 = [0,27,59,91,131,141,163,185,195,275,307,339,357,390,440];
a = a0/max(a0);
d = round(d0*L/d0(15)); % Scale them to fit in length L
d(15)=L;

for i=1:14
       m = d(i):d(i+1)-1;
       slope = (a(i+1)-a(i))/(d(i+1)-d(i));
       x(m+1) = a(i)+slope*(m-d(i));
end

end
```
请参考以上案例生成python程序。

### LLM 生成 ： 
以下是将给定的MATLAB代码转换成Python代码的版本：

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter

# ECG signal generator function
def ecg(L):
    a0 = [0, 1, 40, 1, 0, -34, 118, -99, 0, 2, 21, 2, 0, 0, 0]
    d0 = [0, 27, 59, 91, 131, 141, 163, 185, 195, 275, 307, 339, 357, 390, 440]
    a = np.array(a0) / max(a0)
    d = np.round(np.array(d0) * L / d0[-1]).astype(int)
    d[-1] = L
    x = np.zeros(L)
    
    for i in range(14):
        m = np.arange(d[i], d[i+1])
        slope = (a[i+1] - a[i]) / (d[i+1] - d[i])
        x[m] = a[i] + slope * (m - d[i])
    
    return x

# Generate ECG waveform
wform = ecg(500)

plt.figure()
plt.plot(wform)
plt.axis([0, 500, -1.25, 1.25])
plt.text(155, -0.4, "Q")
plt.text(180, 1.1, "R")
plt.text(205, -1, "S")
plt.title("Original ECG Waveform")
plt.show()

# Add noise to ECG waveform
np.random.seed(0)
x = wform + 0.25 * np.random.randn(500)

# Design a lowpass IIR Butterworth filter
fs = 1000  # Assuming sampling frequency of 1000 Hz
fc = 0.15  # Normalized cutoff frequency
b, a = butter(12, fc, btype='low', analog=False, output='ba')

# Apply zero-phase filtering using filtfilt
y = filtfilt(b, a, x)

# Plot results
plt.figure()
plt.plot(x, label='Noisy ECG')
plt.plot(y, linewidth=3, label='Zero-Phase Filtering')
plt.title('ECG Filtering using Lowpass IIR Butterworth Filter')
plt.legend()
plt.show()

```
 







