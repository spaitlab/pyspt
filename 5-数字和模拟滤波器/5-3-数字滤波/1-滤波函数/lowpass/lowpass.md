# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波

## MATLAB函数描述：lowpass 

函数来源：[MATLAB lowpass](https://ww2.mathworks.cn/help/signal/ref/lowpass.html)

### 语法

y = lowpass(x,wpass)
y = lowpass(x,fpass,fs)
y = lowpass(xt,fpass)
y = lowpass(___,Name=Value)
[y,d] = lowpass(___)
lowpass(___)

### 说明

y = lowpass（x，wpass） 使用低通滤波器滤波输入信号 归一化通带频率，单位为 π rad/sample。 使用 最小阶滤波器，阻带衰减为60 dB，可补偿 滤波器引入的延迟。如果是一个矩阵， 函数独立筛选每一列。xwpasslowpassx

y = lowpass（x，fpass，fs） 指定已以赫兹速率采样。 是通带 滤波器的频率，单位为赫兹。xfsfpass

y = lowpass（xt，fpass） 使用滤波器对时间表中的数据进行低通滤波 通带频率为赫兹。函数 独立过滤时间表中的所有变量以及每个变量中的所有列 变量。xtfpass

y = lowpass(___,Name=Value)使用 name-value 为前面的任何语法指定其他选项 参数。您可以更改阻带衰减、低通滤波器陡度和 滤波器的脉冲响应类型。

[y,d] = lowpass(___)还返回用于筛选输入的 digitalFilter 对象。d

lowpass(___)没有输出参数的图 输入信号并叠加滤波信号。

### 输入参数

x— 输入信号
矢量 |矩阵
输入信号，指定为向量或矩阵。
示例：指定嘈杂的 正弦波sin(2*pi*(0:127)/16)+randn(1,128)/100
示例：指定双通道 正弦波。[2 1].*sin(2*pi*(0:127)'./[16 64])
数据类型： |
复数支持：是singledouble

wpass— 归一化通带频率
标量在 （0， 1）
归一化通带频率，指定为区间 （0， 1） 中的标量。

fpass— 通带频率
标量 in （0， fs/2）
通带频率，指定为区间 （0， fs/2） 中的标量。

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

d— 低通滤波器
digitalFilter 对象
滤波操作中使用的低通滤波器，作为 digitalFilter 对象返回。

使用滤波器滤波信号 x 。与 不同，该函数不补偿滤波器延迟。还可以将 filtfilt 和 fftfilt 函数用于对象。(d,x)dlowpassfilterdigitalFilter

使用过滤器 用于可视化滤波器响应的分析器。

使用 designfilt 进行编辑或 根据频率响应规范生成数字滤波器。

## Python函数描述：lowpass

函数来源：自定义
### 函数定义：
def lowpass(data, cutoff, fs, order=5):
    """
    对给定数据应用低通滤波器。
    
    参数:
    data : array_like
        输入信号数组。
    cutoff : float
        低通滤波器的截止频率 (单位：Hz)。
    fs : float
        采样频率 (单位：Hz)。
    order : int, optional
        滤波器的阶数，默认为5。
    
    返回:
    y : ndarray
        滤波后的信号数组。
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y



## Prompt 1 ： 生成 Python lowpass 函数

参考下面MATLAB代码的lowpass函数

```
% 创建一个以 1 kHz 采样的信号，持续 1 秒。该信号包含两个音调，一个是 50 Hz，另一个是 250 Hz，嵌入方差为 1/100 的高斯白噪声中。高频音调的振幅是低频音调的两倍。对信号进行低通滤波以消除高频音调。指定 150 Hz 的通带频率。 显示原始信号和滤波信号及其频谱。
fs = 1e3;
t = 0:1/fs:1;
x = [1 2]*sin(2*pi*[50 250]'.*t) + randn(size(t))/10;
lowpass(x,150,fs)
```
和我们采用Python语言实现的原始信号和滤波信号及其频谱。
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# 设置采样频率和时间向量
fs = 1000  # 采样频率为 1 kHz
t = np.linspace(0, 1, fs, endpoint=False)  # 持续 1 秒

# 生成信号
frequencies = np.array([50, 250])
amplitudes = np.array([1, 2])
signal = (amplitudes[0] * np.sin(2 * np.pi * frequencies[0] * t) +
          amplitudes[1] * np.sin(2 * np.pi * frequencies[1] * t))

# 添加高斯白噪声
noise = np.random.randn(fs) / 10
x = signal + noise

# 定义低通滤波器函数
def lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# 应用低通滤波器
filtered_signal = lowpass_filter(x, 150, fs, order=5)

# 绘制原始信号和滤波信号及其频谱
plt.figure(figsize=(12, 8))

# 绘制时域信号
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Original Signal with Noise', color='blue')
plt.plot(t, filtered_signal, label='Lowpass Filtered Signal', color='orange', linewidth=2)
plt.title('Time Domain Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.legend()

# 绘制频率响应
plt.subplot(2, 1, 2)
plt.psd(x, Fs=fs, NFFT=512, label='Original Signal PSD', color='blue')
plt.psd(filtered_signal, Fs=fs, NFFT=512, label='Lowpass Filtered Signal PSD', color='orange')
plt.title('Signal Spectra')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density')
plt.legend()

plt.tight_layout()
plt.show()
```
请根据使用上述程序，完成Python语言lowpass函数的编写，便于类似应用调用。


### LLM 生成 ： Python lowpass 函数

```
import numpy as np
from scipy.signal import butter, lfilter

def lowpass(data, cutoff, fs, order=5):
    """
    对给定数据应用低通滤波器。
    
    参数:
    data : array_like
        输入信号数组。
    cutoff : float
        低通滤波器的截止频率 (单位：Hz)。
    fs : float
        采样频率 (单位：Hz)。
    order : int, optional
        滤波器的阶数，默认为5。
    
    返回:
    y : ndarray
        滤波后的信号数组。
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

```
## Prompt 2 ： 应用生成的 Python lowpass 函数做MATLAB示例对应
这是原始程序
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# 设置采样频率和时间向量
fs = 1000  # 采样频率为 1 kHz
t = np.linspace(0, 1, fs, endpoint=False)  # 持续 1 秒

# 生成信号
frequencies = np.array([50, 250])
amplitudes = np.array([1, 2])
signal = (amplitudes[0] * np.sin(2 * np.pi * frequencies[0] * t) +
          amplitudes[1] * np.sin(2 * np.pi * frequencies[1] * t))

# 添加高斯白噪声
noise = np.random.randn(fs) / 10
x = signal + noise

# 定义低通滤波器函数
def lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# 应用低通滤波器
filtered_signal = lowpass_filter(x, 150, fs, order=5)

# 绘制原始信号和滤波信号及其频谱
plt.figure(figsize=(12, 8))

# 绘制时域信号
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Original Signal with Noise', color='blue')
plt.plot(t, filtered_signal, label='Lowpass Filtered Signal', color='orange', linewidth=2)
plt.title('Time Domain Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.legend()

# 绘制频率响应
plt.subplot(2, 1, 2)
plt.psd(x, Fs=fs, NFFT=512, label='Original Signal PSD', color='blue')
plt.psd(filtered_signal, Fs=fs, NFFT=512, label='Lowpass Filtered Signal PSD', color='orange')
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

def lowpass(data, cutoff, fs, order=5):
    """
    对给定数据应用低通滤波器。
    
    参数:
    data : array_like
        输入信号数组。
    cutoff : float
        低通滤波器的截止频率 (单位：Hz)。
    fs : float
        采样频率 (单位：Hz)。
    order : int, optional
        滤波器的阶数，默认为5。
    
    返回:
    y : ndarray
        滤波后的信号数组。
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y




请给出应用函数lowpass求出原始信号和滤波信号及其频谱，给出Python程序。
### LLM 生成 ： 
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# 定义低通滤波器函数
def lowpass(data, cutoff, fs, order=5):
    """
    对给定数据应用低通滤波器。

    参数:
    data : array_like
        输入信号数组。
    cutoff : float
        低通滤波器的截止频率 (单位：Hz)。
    fs : float
        采样频率 (单位：Hz)。
    order : int, optional
        滤波器的阶数，默认为5。

    返回:
    y : ndarray
        滤波后的信号数组。
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# 设置采样频率和时间向量
fs = 1000  # 采样频率为 1 kHz
t = np.linspace(0, 1, fs, endpoint=False)  # 持续 1 秒

# 生成信号
frequencies = np.array([50, 250])
amplitudes = np.array([1, 2])
signal = (amplitudes[0] * np.sin(2 * np.pi * frequencies[0] * t) +
          amplitudes[1] * np.sin(2 * np.pi * frequencies[1] * t))

# 添加高斯白噪声
noise = np.random.randn(fs) / 10
x = signal + noise

# 应用低通滤波器
filtered_signal = lowpass(x, 150, fs, order=5)

# 绘制原始信号和滤波信号及其频谱
plt.figure(figsize=(12, 8))

# 绘制时域信号
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Original Signal with Noise', color='blue')
plt.plot(t, filtered_signal, label='Lowpass Filtered Signal', color='orange', linewidth=2)
plt.title('Time Domain Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.legend()

# 绘制频率响应
plt.subplot(2, 1, 2)
plt.psd(x, Fs=fs, NFFT=512, label='Original Signal PSD', color='blue')
plt.psd(filtered_signal, Fs=fs, NFFT=512, label='Lowpass Filtered Signal PSD', color='orange')
plt.title('Signal Spectra')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density')
plt.legend()

plt.tight_layout()
plt.show()
```

## Prompt 3 ： 应用生成的 Python lowpass.py 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：


**Matlab代码**
```
% 创建一个以 1 kHz 采样的信号，持续 1 秒。该信号包含两个音调，一个是 50 Hz，另一个是 250 Hz，嵌入方差为 1/100 的高斯白噪声中。高频音调的振幅是低频音调的两倍。对信号进行低通滤波以消除高频音调。指定 150 Hz 的通带频率。 显示原始信号和滤波信号及其频谱。
fs = 1e3;
t = 0:1/fs:1;
x = [1 2]*sin(2*pi*[50 250]'.*t) + randn(size(t))/10;
lowpass(x,150,fs)
```

这是Python lowpass.py
```
import numpy as np
from scipy.signal import butter, lfilter

def lowpass(data, cutoff, fs, order=5):
    """
    对给定数据应用低通滤波器。
    
    参数:
    data : array_like
        输入信号数组。
    cutoff : float
        低通滤波器的截止频率 (单位：Hz)。
    fs : float
        采样频率 (单位：Hz)。
    order : int, optional
        滤波器的阶数，默认为5。
    
    返回:
    y : ndarray
        滤波后的信号数组。
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


```


### LLM 生成 ： 

下面这是调用程序
```
import numpy as np
import matplotlib.pyplot as plt
from lowpass import lowpass

# 设置采样频率和时间向量
fs = 1000  # 采样频率为 1 kHz
t = np.linspace(0, 1, fs, endpoint=True)  # 持续 1 秒，包含终点

# 生成信号
frequencies = np.array([50, 250])
amplitudes = np.array([1, 2])
signal = (amplitudes[0] * np.sin(2 * np.pi * frequencies[0] * t) +
          amplitudes[1] * np.sin(2 * np.pi * frequencies[1] * t))

# 添加高斯白噪声
noise = np.random.randn(len(t)) / 10
x = signal + noise

# 应用低通滤波器
filtered_signal = lowpass(x, 150, fs, order=5)

# 绘制原始信号和滤波信号及其频谱
plt.figure(figsize=(12, 8))

# 绘制时域信号
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Original Signal with Noise', color='blue')
plt.plot(t, filtered_signal, label='Lowpass Filtered Signal', color='orange', linewidth=2)
plt.title('Time Domain Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.legend()

# 绘制频率响应
plt.subplot(2, 1, 2)
plt.psd(x, Fs=fs, NFFT=512, label='Original Signal PSD', color='blue')
plt.psd(filtered_signal, Fs=fs, NFFT=512, label='Lowpass Filtered Signal PSD', color='orange')
plt.title('Signal Spectra')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density')
plt.legend()

plt.tight_layout()
plt.show()


```