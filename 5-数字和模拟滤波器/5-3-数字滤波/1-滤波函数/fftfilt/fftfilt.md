# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波

## MATLAB函数描述：fftfilt 

函数来源：[MATLAB fftfilt](https://ww2.mathworks.cn/help/signal/ref/fftfilt.html)

### 语法

y = fftfilt(b,x)
y = fftfilt(b,x,n)
y = fftfilt(d,x)
y = fftfilt(d,x,n)

### 说明

y = fftfilt（b，x） 使用系数描述的过滤器过滤向量中的数据 向量。xb

y = fftfilt（b，x，n） 用于确定 FFT 的长度。n

y = fftfilt（d，x） 使用 digitalFilter 对象过滤 vector 中的数据。xd

y = fftfilt（d，x，n） 用于确定 FFT 的长度。n

### 输入参数

b— 滤波器系数
向量 |矩阵
滤波器系数，指定为向量。如果是一个矩阵，则将每一列中的滤波器应用于信号向量 x。bfftfiltb

x— 输入数据
向量 |矩阵
输入数据，指定为向量。如果是矩阵，则过滤其列。如果 b 和 都是具有相同列数的矩阵，则 的第 i列用于过滤 的第 i列。 适用于实际输入和复杂输入。xfftfiltxbxfftfilt

n— FFT 长度
正整数
FFT 长度，指定为正整数。默认情况下，选择 FFT 长度和数据块长度 保证高效的执行时间。fftfilt

d— 数字滤波器
digitalFilter 对象
数字筛选器，指定为 digitalFilter 对象。使用 designfilt 根据频率响应生成 规格。d


### 输出参量

y— 输出数据
向量 | 矩阵
输出数据，以向量或矩阵形式返回。

## Python函数描述：fftfilt

函数来源：自定义
### 函数定义：
def fftfilt(b, x):
    """
    使用 FFT 在 GPU 上应用 FIR 滤波器。
    
    参数:
    b : array_like
        FIR 滤波器的系数（一维数组）。
    x : array_like
        要滤波的信号（一维数组）。
    
    返回:
    y : ndarray
        滤波后的信号数组。
    """
    # 将数据和滤波器系数传输到 GPU
    x_gpu = cp.asarray(x)
    b_gpu = cp.asarray(b)

    # 使用 FFT 进行滤波 (频域卷积)
    y_gpu = cp.fft.ifft(cp.fft.fft(x_gpu) * cp.fft.fft(b_gpu, n=len(x_gpu)))
    
    # 将结果转换回 CPU 并取实部
    y = cp.asnumpy(y_gpu).real
    
    return y


## Prompt 1 ： 生成 Python fftfilt 函数

参考下面MATLAB代码的fftfilt函数

```
% 创建一个由白高斯加法噪声中的正弦波总和组成的信号。正弦波频率为 2.5、5、10 和 15 kHz。采样频率为 50 kHz。使用设计低通 FIR 等纹波滤波器。使用 overlap-add 方法过滤 GPU 上的数据。使用 将数据放在 GPU 上。使用并绘制滤波数据的功率谱密度估计值，将输出返回到 MATLAB® 工作区。
Fs = 50e3;
t = 0:1/Fs:10-(1/Fs);
x = cos(2*pi*2500*t) + 0.5*sin(2*pi*5000*t) + 0.25*cos(2*pi*10000*t)+ ...
    0.125*sin(2*pi*15000*t) + randn(size(t));
d = designfilt('lowpassfir','SampleRate',Fs, ...
    'PassbandFrequency',5500,'StopbandFrequency',6000, ...
    'PassbandRipple',0.5,'StopbandAttenuation',50);
B = d.Coefficients;
y = fftfilt(gpuArray(B),gpuArray(x));
periodogram(gather(y),rectwin(length(y)),length(y),50e3)
```
和我们采用Python语言实现的滤波数据的功率谱密度估计值。
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, fftconvolve, periodogram
import cupy as cp  # 使用 CuPy 作为 GPU 加速库

# 设置采样参数
Fs = 50000  # 采样频率 50 kHz
t = np.arange(0, 10, 1/Fs)  # 时间向量

# 生成信号
x = (np.cos(2 * np.pi * 2500 * t) + 
     0.5 * np.sin(2 * np.pi * 5000 * t) + 
     0.25 * np.cos(2 * np.pi * 10000 * t) + 
     0.125 * np.sin(2 * np.pi * 15000 * t) + 
     np.random.randn(*t.shape))  # 加入白高斯噪声

# 设计低通 FIR 滤波器
numtaps = 101  # 滤波器阶数
cutoff = 5500  # 通带截止频率
B = firwin(numtaps, cutoff, fs=Fs, window="hamming")  # 使用 Hamming 窗

# 将数据和滤波器系数传输到 GPU
x_gpu = cp.asarray(x)
B_gpu = cp.asarray(B)

# 使用 FFT 进行滤波 (频域卷积)
y_gpu = cp.fft.ifft(cp.fft.fft(x_gpu) * cp.fft.fft(B_gpu, n=len(x_gpu)))
y = cp.asnumpy(y_gpu).real  # 将结果转换回 CPU 并取实部

# 绘制功率谱密度估计
f, Pxx_den = periodogram(y, Fs, window="rectangular", scaling='density')
plt.figure(figsize=(10, 6))
plt.semilogy(f, Pxx_den)
plt.title('Power Spectral Density of Filtered Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.grid(True)
plt.show()
```
请根据使用上述程序，完成Python语言fftfilt函数的编写，便于类似应用调用。


### LLM 生成 ： Python fftfilt 函数

```
import numpy as np
import cupy as cp

def fftfilt(b, x):
    """
    使用 FFT 在 GPU 上应用 FIR 滤波器。
    
    参数:
    b : array_like
        FIR 滤波器的系数（一维数组）。
    x : array_like
        要滤波的信号（一维数组）。
    
    返回:
    y : ndarray
        滤波后的信号数组。
    """
    # 将数据和滤波器系数传输到 GPU
    x_gpu = cp.asarray(x)
    b_gpu = cp.asarray(b)

    # 使用 FFT 进行滤波 (频域卷积)
    y_gpu = cp.fft.ifft(cp.fft.fft(x_gpu) * cp.fft.fft(b_gpu, n=len(x_gpu)))
    
    # 将结果转换回 CPU 并取实部
    y = cp.asnumpy(y_gpu).real
    
    return y
```
## Prompt 2 ： 应用生成的 Python fftfilt 函数做MATLAB示例对应
这是原始程序
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, fftconvolve, periodogram
import cupy as cp  # 使用 CuPy 作为 GPU 加速库

# 设置采样参数
Fs = 50000  # 采样频率 50 kHz
t = np.arange(0, 10, 1/Fs)  # 时间向量

# 生成信号
x = (np.cos(2 * np.pi * 2500 * t) + 
     0.5 * np.sin(2 * np.pi * 5000 * t) + 
     0.25 * np.cos(2 * np.pi * 10000 * t) + 
     0.125 * np.sin(2 * np.pi * 15000 * t) + 
     np.random.randn(*t.shape))  # 加入白高斯噪声

# 设计低通 FIR 滤波器
numtaps = 101  # 滤波器阶数
cutoff = 5500  # 通带截止频率
B = firwin(numtaps, cutoff, fs=Fs, window="hamming")  # 使用 Hamming 窗

# 将数据和滤波器系数传输到 GPU
x_gpu = cp.asarray(x)
B_gpu = cp.asarray(B)

# 使用 FFT 进行滤波 (频域卷积)
y_gpu = cp.fft.ifft(cp.fft.fft(x_gpu) * cp.fft.fft(B_gpu, n=len(x_gpu)))
y = cp.asnumpy(y_gpu).real  # 将结果转换回 CPU 并取实部

# 绘制功率谱密度估计
f, Pxx_den = periodogram(y, Fs, window="rectangular", scaling='density')
plt.figure(figsize=(10, 6))
plt.semilogy(f, Pxx_den)
plt.title('Power Spectral Density of Filtered Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.grid(True)
plt.show()
```

这是函数定义
import numpy as np
import cupy as cp

def fftfilt(b, x):
    """
    使用 FFT 在 GPU 上应用 FIR 滤波器。
    
    参数:
    b : array_like
        FIR 滤波器的系数（一维数组）。
    x : array_like
        要滤波的信号（一维数组）。
    
    返回:
    y : ndarray
        滤波后的信号数组。
    """
    # 将数据和滤波器系数传输到 GPU
    x_gpu = cp.asarray(x)
    b_gpu = cp.asarray(b)

    # 使用 FFT 进行滤波 (频域卷积)
    y_gpu = cp.fft.ifft(cp.fft.fft(x_gpu) * cp.fft.fft(b_gpu, n=len(x_gpu)))
    
    # 将结果转换回 CPU 并取实部
    y = cp.asnumpy(y_gpu).real
    
    return y

请给出应用函数fftfilt求出滤波数据的功率谱密度估计值，给出Python程序。
### LLM 生成 ： 
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def bandpass(data, freq_band, fs, order=5):
    """
    对给定数据应用带通滤波器。

    参数:
    data : array_like
        输入信号数组。
    freq_band : array_like
        两个元素的列表或数组，定义了滤波器的通带，即 [低频截止, 高频截止] (单位：Hz)。
    fs : float
        采样频率 (单位：Hz)。
    order : int, optional
        滤波器的阶数，默认为5。

    返回:
    y : ndarray
        滤波后的信号数组。
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = freq_band[0] / nyq
    high = freq_band[1] / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

# 设置采样频率和时间向量
fs = 1000
t = np.arange(0, 1, 1/fs)

# 生成信号
frequencies = np.array([50, 150, 250])
amplitudes = np.array([2, 1, 2])
signal = (amplitudes[0] * np.sin(2 * np.pi * frequencies[0] * t) +
          amplitudes[1] * np.sin(2 * np.pi * frequencies[1] * t) +
          amplitudes[2] * np.sin(2 * np.pi * frequencies[2] * t))

# 添加高斯白噪声
noise = np.random.randn(len(t)) / 10
x = signal + noise

# 使用自定义 bandpass 函数应用带通滤波器
filtered_signal = bandpass(x, [100, 200], fs)

# 绘制原始信号和滤波信号
plt.figure(figsize=(12, 8))

# 绘制时域信号
plt.subplot(2, 2, 1)
plt.plot(t, x, label='Original Signal with Noise')
plt.plot(t, filtered_signal, label='Bandpass Filtered Signal', linewidth=2)
plt.title('Time Domain Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.legend()

# 绘制时域信号的频谱
plt.subplot(2, 2, 2)
plt.psd(x, Fs=fs, NFFT=512, label='Original Signal PSD', color='blue')
plt.psd(filtered_signal, Fs=fs, NFFT=512, label='Filtered Signal PSD', color='orange')
plt.title('Signal Spectra')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density')
plt.legend()

plt.tight_layout()
plt.show()
```

## Prompt 3 ： 应用生成的 Python fftfilt.py 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：


**Matlab代码**
```
% 创建一个由白高斯加法噪声中的正弦波总和组成的信号。正弦波频率为 2.5、5、10 和 15 kHz。采样频率为 50 kHz。使用设计低通 FIR 等纹波滤波器。使用 overlap-add 方法过滤 GPU 上的数据。使用 将数据放在 GPU 上。使用并绘制滤波数据的功率谱密度估计值，将输出返回到 MATLAB® 工作区。
Fs = 50e3;
t = 0:1/Fs:10-(1/Fs);
x = cos(2*pi*2500*t) + 0.5*sin(2*pi*5000*t) + 0.25*cos(2*pi*10000*t)+ ...
    0.125*sin(2*pi*15000*t) + randn(size(t));
d = designfilt('lowpassfir','SampleRate',Fs, ...
    'PassbandFrequency',5500,'StopbandFrequency',6000, ...
    'PassbandRipple',0.5,'StopbandAttenuation',50);
B = d.Coefficients;
y = fftfilt(gpuArray(B),gpuArray(x));
periodogram(gather(y),rectwin(length(y)),length(y),50e3)
```

这是Python fftfilt.py
```
import numpy as np
import cupy as cp

def fftfilt(b, x):
    """
    使用 FFT 在 GPU 上应用 FIR 滤波器。
    
    参数:
    b : array_like
        FIR 滤波器的系数（一维数组）。
    x : array_like
        要滤波的信号（一维数组）。
    
    返回:
    y : ndarray
        滤波后的信号数组。
    """
    # 将数据和滤波器系数传输到 GPU
    x_gpu = cp.asarray(x)
    b_gpu = cp.asarray(b)

    # 使用 FFT 进行滤波 (频域卷积)
    y_gpu = cp.fft.ifft(cp.fft.fft(x_gpu) * cp.fft.fft(b_gpu, n=len(x_gpu)))
    
    # 将结果转换回 CPU 并取实部
    y = cp.asnumpy(y_gpu).real
    
    return y
```


### LLM 生成 ： 

下面这是调用程序
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, periodogram
from fftfilt import fftfilt  # 导入我们定义的 fftfilt 函数

# 设置采样参数
Fs = 50000  # 采样频率 50 kHz
t = np.arange(0, 10, 1/Fs)  # 时间向量，10秒长

# 生成信号
x = (np.cos(2 * np.pi * 2500 * t) + 
     0.5 * np.sin(2 * np.pi * 5000 * t) + 
     0.25 * np.cos(2 * np.pi * 10000 * t) + 
     0.125 * np.sin(2 * np.pi * 15000 * t) + 
     np.random.randn(*t.shape))  # 加入白高斯噪声

# 设计低通 FIR 滤波器
numtaps = 101  # 滤波器阶数
cutoff = 5500  # 通带截止频率
B = firwin(numtaps, cutoff, fs=Fs, window="hamming")  # 使用 Hamming 窗

# 使用 fftfilt 进行滤波
y = fftfilt(B, x)

# 绘制功率谱密度估计
f, Pxx_den = periodogram(y, Fs, window="rectangular", scaling='density')
plt.figure(figsize=(10, 6))
plt.semilogy(f, Pxx_den)
plt.title('Power Spectral Density of Filtered Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.grid(True)
plt.show()
```