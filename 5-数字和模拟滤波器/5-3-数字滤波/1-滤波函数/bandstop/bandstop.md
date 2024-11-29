# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波

## MATLAB函数描述：bandstop 

函数来源：[MATLAB bandstop](https://ww2.mathworks.cn/help/signal/ref/bandstop.html)

### 语法

y = bandstop(x,wpass)
y = bandstop(x,fpass,fs)
y = bandstop(xt,fpass)
y = bandstop(___,Name=Value)
[y,d] = bandstop(___)
bandstop(___)

### 说明

y = bandstop（x，wpass） 使用带阻滤波器滤波输入信号，并带有 由双元素矢量指定的阻带频率范围，以 π rad/sample 的归一化单位表示。 使用 最小阶滤波器，阻带衰减为60 dB，可补偿 滤波器引入的延迟。如果是一个矩阵， 函数独立筛选每一列。xwpassbandstopx

y = bandstop（x，fpass，fs） 指定已以赫兹速率采样。双元素矢量指定 以赫兹为单位进行过滤。xfsfpass

y = bandstop（xt，fpass） bandstop-使用过滤器过滤时间表中的数据 阻带频率范围由双元素矢量指定，单位为赫兹。该函数独立过滤所有变量 在时间表和每个变量内的所有列中。xtfpass

y = bandstop(___,Name=Value)使用 name-value 为前面的任何语法指定其他选项 参数。您可以更改阻带衰减、带阻滤波器陡峭度和 滤波器的脉冲响应类型。

[y,d] = bandstop(___)还返回用于筛选输入的 digitalFilter 对象。d

bandstop(___)没有输出参数的图 输入信号并叠加滤波信号。

### 输入参数

x— 输入信号
矢量 |矩阵
输入信号，指定为向量或矩阵。
示例：指定嘈杂的 正弦波sin(2*pi*(0:127)/16)+randn(1,128)/100
示例：指定双通道 正弦波。[2 1].*sin(2*pi*(0:127)'./[16 64])
数据类型： |
复数支持：是singledouble

wpass— 元素在 （0， 1） 中的归一化阻带频率范围
双元素向量
归一化阻带频率范围，指定为双元素矢量 元素在区间 （0， 1） 中。

fpass— 阻带频率范围
双元素矢量，元素在 （0， fs/2）
阻带频率范围，指定为带元素的双元素向量 在区间 （0， fs/2） 中。

fs— 采样率
正实标量
采样率，指定为正实数标量。

xt— 输入时间表
时间表
输入时间表。 必须包含递增的、有限的和等距的 持续时间类型的行时间（以秒为单位）。xt
如果时间表缺少或重复的时间点，您可以使用“使用缺失、重复或不均匀时间清理时间表”中的提示进行修复。
示例：包含一个 单通道随机信号和双通道随机信号，以 1 Hz 采样 4 秒。timetable(seconds(0:4)',randn(5,1),randn(5,2))
示例：包含一个单通道随机信号和一个双通道随机信号 信号，以 1 Hz 采样 4 秒。timetable(randn(5,1),randn(5,2),SampleRate=1)

### 输出参量

y— 滤波信号
矢量 | 矩阵 | 时间表
滤波信号，以向量、矩阵或时间表的形式返回，与输入的维度相同。

d— 带阻滤波器
digitalFilter 对象
过滤操作中使用的带阻滤波器，作为 digitalFilter 对象返回。
使用过滤器过滤 信号 x 使用 。 与 不同，该函数不补偿 滤波器延迟。您还可以使用 filtfilt 和 fftfilt 函数 与对象。(d,x)dbandstopfilterdigitalFilter
使用过滤器分析器可视化过滤器 响应。
使用 designfilt 可以 根据频率响应编辑或生成数字滤波器 规格。

## Python函数描述：bandstop

函数来源：自定义

### 函数定义：
def bandstop(data, freq_band, fs, order=5):
    """
    Apply a bandstop filter to the data.
    
    Parameters:
    data : array_like
        Input signal array.
    freq_band : array_like
        List or array of two elements defining the stop band frequencies, [low cut, high cut] in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the filter, default is 5.
    
    Returns:
    y : ndarray
        Filtered signal array.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    low = freq_band[0] / nyq
    high = freq_band[1] / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y


## Prompt 1 ： 生成 Python bandstop 函数


参考下面MATLAB代码的bandstop函数

```
% 创建一个以 1 kHz 采样的信号，持续 1 秒。该信号包含三个音调，一个为 50 Hz，另一个为 150 Hz，第三个为 250 Hz。高频和低频音调的振幅都是中间音调的两倍。信号嵌入方差为1/100的高斯白噪声中。对信号进行带通滤波以去除低频和高频音调。指定 100 Hz 和 200 Hz 的通带频率。 显示原始信号和滤波信号及其频谱。
fs = 1e3;
t = 0:1/fs:1;
x = [2 1 2]*sin(2*pi*[50 150 250]'.*t) + randn(size(t))/10;
bandstop(x,[100 200],fs)
```
和我们采用Python语言实现的原始信号和滤波信号及其频谱。
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# 设置采样频率和时间向量
fs = 1000  # 采样频率为1 kHz
t = np.linspace(0, 1, fs, endpoint=False)  # 持续1秒

# 生成信号
frequencies = np.array([50, 150, 250])
amplitudes = np.array([2, 1, 2])
signal = (amplitudes[0] * np.sin(2 * np.pi * frequencies[0] * t) +
          amplitudes[1] * np.sin(2 * np.pi * frequencies[1] * t) +
          amplitudes[2] * np.sin(2 * np.pi * frequencies[2] * t))

# 添加高斯白噪声
noise = np.random.randn(fs) / 10
x = signal + noise

# 定义带阻滤波器函数
def bandstop_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y

# 应用带阻滤波器
filtered_signal = bandstop_filter(x, 100, 200, fs, order=5)

# 绘制原始信号和滤波信号及其频谱
plt.figure(figsize=(12, 8))

# 绘制时域信号
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Original Signal with Noise', color='blue')
plt.plot(t, filtered_signal, label='Bandstop Filtered Signal', color='orange', linewidth=2)
plt.title('Time Domain Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.legend()

# 绘制频率响应
plt.subplot(2, 1, 2)
plt.psd(x, Fs=fs, NFFT=512, label='Original Signal PSD', color='blue')
plt.psd(filtered_signal, Fs=fs, NFFT=512, label='Filtered Signal PSD', color='orange')
plt.title('Signal Spectra')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density')
plt.legend()

plt.tight_layout()
plt.show()
```
请根据使用上述程序，完成Python语言bandstop函数的编写，便于类似应用调用。


### LLM 生成 ： Python bandstop 函数

```
import numpy as np
from scipy.signal import butter, lfilter

def bandstop(data, freq_band, fs, order=5):
    """
    Apply a bandstop filter to the data.
    
    Parameters:
    data : array_like
        Input signal array.
    freq_band : array_like
        List or array of two elements defining the stop band frequencies, [low cut, high cut] in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the filter, default is 5.
    
    Returns:
    y : ndarray
        Filtered signal array.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    low = freq_band[0] / nyq
    high = freq_band[1] / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y

```
## Prompt 2 ： 应用生成的 Python bandstop 函数做MATLAB示例对应
这是原始程序
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# 设置采样频率和时间向量
fs = 1000  # 采样频率为1 kHz
t = np.linspace(0, 1, fs, endpoint=False)  # 持续1秒

# 生成信号
frequencies = np.array([50, 150, 250])
amplitudes = np.array([2, 1, 2])
signal = (amplitudes[0] * np.sin(2 * np.pi * frequencies[0] * t) +
          amplitudes[1] * np.sin(2 * np.pi * frequencies[1] * t) +
          amplitudes[2] * np.sin(2 * np.pi * frequencies[2] * t))

# 添加高斯白噪声
noise = np.random.randn(fs) / 10
x = signal + noise

# 定义带阻滤波器函数
def bandstop_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y

# 应用带阻滤波器
filtered_signal = bandstop_filter(x, 100, 200, fs, order=5)

# 绘制原始信号和滤波信号及其频谱
plt.figure(figsize=(12, 8))

# 绘制时域信号
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Original Signal with Noise', color='blue')
plt.plot(t, filtered_signal, label='Bandstop Filtered Signal', color='orange', linewidth=2)
plt.title('Time Domain Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.legend()

# 绘制频率响应
plt.subplot(2, 1, 2)
plt.psd(x, Fs=fs, NFFT=512, label='Original Signal PSD', color='blue')
plt.psd(filtered_signal, Fs=fs, NFFT=512, label='Filtered Signal PSD', color='orange')
plt.title('Signal Spectra')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density')
plt.legend()

plt.tight_layout()
plt.show()
```

这是函数定义
import numpy as np
from scipy.signal import butter, lfilter

def bandstop(data, freq_band, fs, order=5):
    """
    Apply a bandstop filter to the data.
    
    Parameters:
    data : array_like
        Input signal array.
    freq_band : array_like
        List or array of two elements defining the stop band frequencies, [low cut, high cut] in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the filter, default is 5.
    
    Returns:
    y : ndarray
        Filtered signal array.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    low = freq_band[0] / nyq
    high = freq_band[1] / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y


请给出应用函数bandstop求出原始信号和滤波信号及其频谱，给出Python程序。
### LLM 生成 ： 
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def bandstop(data, freq_band, fs, order=5):
    """
    Apply a bandstop filter to the data.
    
    Parameters:
    data : array_like
        Input signal array.
    freq_band : array_like
        List or array of two elements defining the stop band frequencies, [low cut, high cut] in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the filter, default is 5.
    
    Returns:
    y : ndarray
        Filtered signal array.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    low = freq_band[0] / nyq
    high = freq_band[1] / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y

# Settings
fs = 1000  # Sampling frequency: 1 kHz
t = np.linspace(0, 1, fs, endpoint=False)  # Duration: 1 second

# Generate the signal with three tones: 50 Hz, 150 Hz, and 250 Hz
frequencies = np.array([50, 150, 250])
amplitudes = np.array([2, 1, 2])
signal = np.sum([amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t) for i in range(len(frequencies))], axis=0)

# Add Gaussian white noise
noise = np.random.randn(fs) / 10
x = signal + noise

# Apply the bandstop filter
filtered_signal = bandstop(x, [100, 200], fs, order=5)

# Plotting
plt.figure(figsize=(12, 8))

# Time domain signals
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Original Signal with Noise', color='blue')
plt.plot(t, filtered_signal, label='Bandstop Filtered Signal', color='orange', linewidth=2)
plt.title('Time Domain Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.legend()

# Frequency response (Power Spectral Density)
plt.subplot(2, 1, 2)
plt.psd(x, Fs=fs, NFFT=512, label='Original Signal PSD', color='blue')
plt.psd(filtered_signal, Fs=fs, NFFT=512, label='Filtered Signal PSD', color='orange')
plt.title('Signal Spectra')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density')
plt.legend()

plt.tight_layout()
plt.show()
```

## Prompt 3 ： 应用生成的 Python bandstop.py 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：
**Matlab代码**
```
% 创建一个以 1 kHz 采样的信号，持续 1 秒。该信号包含三个音调，一个为 50 Hz，另一个为 150 Hz，第三个为 250 Hz。高频和低频音调的振幅都是中间音调的两倍。信号嵌入方差为1/100的高斯白噪声中。对信号进行带通滤波以去除低频和高频音调。指定 100 Hz 和 200 Hz 的通带频率。 显示原始信号和滤波信号及其频谱。
fs = 1e3;
t = 0:1/fs:1;
x = [2 1 2]*sin(2*pi*[50 150 250]'.*t) + randn(size(t))/10;
bandstop(x,[100 200],fs)
```

这是Python bandstop.py
```
import numpy as np
from scipy.signal import butter, lfilter

def bandstop(data, freq_band, fs, order=5):
    """
    Apply a bandstop filter to the data.
    
    Parameters:
    data : array_like
        Input signal array.
    freq_band : array_like
        List or array of two elements defining the stop band frequencies, [low cut, high cut] in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the filter, default is 5.
    
    Returns:
    y : ndarray
        Filtered signal array.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    low = freq_band[0] / nyq
    high = freq_band[1] / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y
```


### LLM 生成 ： 

下面这是调用程序
```
import numpy as np
import matplotlib.pyplot as plt
from bandpass import bandpass

# Define sampling frequency and create time vector
fs = 1000  # Sampling frequency: 1000 Hz
t = np.arange(0, 1, 1/fs)  # Time vector from 0 to 1 second

# Generate the signal: contains three tones 50 Hz, 150 Hz, 250 Hz
frequencies = np.array([50, 150, 250])
amplitudes = np.array([2, 1, 2])
signal = (amplitudes * np.sin(2 * np.pi * frequencies[:, None] * t)).sum(axis=0)

# Add Gaussian white noise
noise = np.random.randn(len(t)) / 10
x = signal + noise

# Apply bandpass filter
filtered_signal = bandpass(x, [100, 200], fs)

# Save the original and filtered signals to files
np.savetxt('original_signal.csv', x, delimiter=',')
np.savetxt('filtered_signal.csv', filtered_signal, delimiter=',')

# Load signals from files
x_loaded = np.loadtxt('original_signal.csv', delimiter=',')
filtered_signal_loaded = np.loadtxt('filtered_signal.csv', delimiter=',')

# Plotting
plt.figure(figsize=(12, 6))

# Time domain signals
plt.subplot(1, 2, 1)
plt.plot(t, x_loaded, label='Original Signal with Noise', alpha=0.5)
plt.plot(t, filtered_signal_loaded, label='Bandpass Filtered Signal', linewidth=2)
plt.title('Time Domain Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.legend()

# Frequency spectrum
plt.subplot(1, 2, 2)
plt.magnitude_spectrum(x_loaded, Fs=fs, scale='dB', color='blue', label='Original Signal Spectrum')
plt.magnitude_spectrum(filtered_signal_loaded, Fs=fs, scale='dB', color='orange', label='Filtered Signal Spectrum')
plt.title('Frequency Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.legend()

plt.tight_layout()
plt.show()
```