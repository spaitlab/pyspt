# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：firpmord

函数来源：[MATLAB firpmord](https://ww2.mathworks.cn/help/signal/ref/firpmord.html)

### 语法

[n,fo,ao,w] = firpmord(f,a,dev)
[___ ] = firpmord(___,fs)
c = firpmord(___,'cell')

### 说明

[n，fo，ao，w] = firpmord（f，a，dev） 返回满足输入规范 、和的近似阶数、归一化频带边、频带幅度和权重。
[___ ]= firpmord(___,fs)指定采样频率。默认值为2Hz，表示奈奎斯特频率为 1 Hz。您可以指定缩放为特定应用程序的采样率。您可以将其与前面的任何输入一起使用语法。
c = firpmord(___,'cell')返回一个元胞数组，其元素是 FIRPM 的参数。

### 输入参数

f— 频带边
沿实值向量
频带边，指定为实值向量。参数必须位于范围 [0，Fs/2]，其中Fs是奈奎斯特频率。数量向量中的元素始终是 2 的倍数。频率必须增加次序。

a— 所需振幅
矢量
f 中包含的点处的所需振幅，指定为向量。并且必须满足条件 length（f） = 2长度（a）–2。所需的函数是分段常数。

dev— 最大允许偏差
矢量
最大允许偏差，指定为向量。 具有与 A 的大小相同。它指定最大允许偏差或 频率响应与输出滤波器所需幅度之间的纹波 对于每个波段。

fs— 采样率
2 Hz（默认）|实标量
采样率，指定为实数标量。

### 输出参量

n— 筛选顺序
为正整数
筛选顺序，以正整数形式返回。

fo— 归一化频率点
实值向量
归一化频率点，指定为实值向量。参数必须是在 [0， 1] 范围内，其中 1 对应于奈奎斯特频率。数量向量中的元素始终是 2 的倍数。频率必须增加次序。

ao— 幅度响应
实值向量
振幅响应，以实值向量形式返回。

w— 权重
实值向量
用于调整每个频带拟合的权重，指定为实值向量。的长度是 f 和 a 长度的一半，所以每个波段只有一个权重。

c— FIR 滤波器参数
单元阵列
FIR 滤波器参数，以元胞数组的形式返回。


## Python函数描述：firpmord

函数来源：自定义

### 函数定义：

def firpmord(f, a, dev, fs):
    """
    Estimate the minimum order for a Parks-McClellan (firpm) FIR filter.
    
    Parameters:
        f (array_like): Cutoff frequencies.
        a (array_like): Desired amplitudes.
        dev (array_like): Maximum passband and minimum stopband ripple.
        fs (float): Sampling frequency.
    
    Returns:
        n (int): Minimum filter order.
        fo (ndarray): Optimal frequencies.
        ao (ndarray): Optimal amplitudes.
        w (ndarray): Optimal weights.
    """
    assert len(f) == len(a) == len(dev) == 2, "Input arrays must have length 2."
    
    rp, rs = dev
    delta_p = (10 ** (rp / 20) - 1) / (10 ** (rp / 20) + 1)
    delta_s = 10 ** (-rs / 20)
    
    def objective(n):
        delta = np.array([delta_p, delta_s])
        m = (n - 1) // 2
        k = np.arange(1, m + 1)
        h = np.sinc(f[1] * k / fs) - np.sinc(f[0] * k / fs)
        a_m = 2 * np.sum(np.cos(2 * np.pi * f[1] * k / fs) * h)
        d_m = 2 * np.sum(np.cos(2 * np.pi * f[1] * k / fs) * np.sin(np.pi * k / m) / (np.pi * k))
        d_m += np.sum(np.cos(2 * np.pi * f[0] * k / fs) * np.sin(np.pi * k / m) / (np.pi * k))
        
        # Modify alpha calculation to ensure shape compatibility
        alpha = np.sum(np.abs(1 + (-1) ** k * a_m))
        delta_a = alpha - 1
        delta_d = (1 - d_m) / 2
        return delta @ np.array([delta_a, delta_d])
    
    res = minimize_scalar(objective, method='bounded', bounds=(1, 1000))
    n = int(np.ceil(res.x))
    
    m = (n - 1) // 2
    fo = np.array([0, f[0] / fs, f[1] / fs, 0.5])
    ao = np.array([a[0], a[0], a[1], a[1]])
    w = np.array([1, delta_p / delta_s, 1])
    
    return n, fo, ao, w

### 参数

- `f （array_like）`：截止频率。
- `a （array_like）`：所需振幅。
- `dev（array_like）`：最大通带和最小阻带纹波。
- `fs（float）`：采样频率。

### 返回值

- `n （int）`：最小筛选顺序。
- `fo （ndarray）`：最佳频率。
- `ao （ndarray）`：最佳振幅。
- `w （ndarray）`：最佳权重。

### 函数工作原理

该函数实现了 Parks-McClellan 算法用于设计滤波器。这个算法的主要思想是在频域上最小化实际响应与期望响应之间的误差，通过迭代优化滤波器系数以达到最小化这种误差的目标。

### 使用场景

1.滤波器设计: 适用于需要指定截止频率、通带最大波纹和阻带最小波纹的数字滤波器设计。常见应用包括音频处理、通信系统等。
2.频域分析: 可以用于分析滤波器的频率响应特性，比如查看滤波器的通带、阻带、过渡带等特性。

### 改进建议

- 参数灵活性提升: 添加更多参数选项，如窗函数类型、滤波器类型等，以增加函数的适用范围和灵活性。
- 文档增强: 提供更详细的文档说明，解释参数的作用和函数的使用方法，以便用户更好地理解如何使用该函数进行滤波器设计。
- 算法优化: 考虑优化算法的实现，以提高设计滤波器的效率和性能，特别是对于大型滤波器设计问题，可能需要更快的优化算法来加速计算过程。



## Prompt 1 ： 生成 Python firpmord 函数

参考下面MATLAB代码的firpmord函数
```
rp = 3;           % Passband ripple in dB 
rs = 40;          % Stopband ripple in dB
fs = 2000;        % Sampling frequency
f = [500 600];    % Cutoff frequencies
a = [1 0];        % Desired amplitudes

dev = [(10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)]; 
[n,fo,ao,w] = firpmord(f,a,dev,fs);
b = firpm(n,fo,ao,w);
freqz(b,1,1024,fs)
title('Lowpass Filter Designed to Specifications')
```

和我们采用Python语言实现的最小阶低通滤波器，
```
import numpy as np
import matplotlib.pyplot as plt

# 指定参数
rp = 3           # Passband ripple in dB 
rs = 40          # Stopband ripple in dB
fs = 2000        # Sampling frequency
f_pass = 500     # Passband cutoff frequency in Hz
f_stop = 600     # Stopband cutoff frequency in Hz

# 将通带和阻带波纹转换为线性单位
dev_pass = (10**(rp / 20) - 1) / (10**(rp / 20) + 1)
dev_stop = 10**(-rs / 20)

# 计算滤波器阶数
delta_omega = 2 * np.pi * (f_stop - f_pass) / fs
n = int(np.ceil((rs - 8) / (2.285 * delta_omega)))

# 计算窗函数
h = np.hamming(n + 1)

# 计算频率响应
omega = np.linspace(0, np.pi, 1000)
h_freqz = np.fft.fft(h, 1000)
H = np.abs(h_freqz)
H_db = 20 * np.log10(H)

# 绘制幅度响应
plt.figure(figsize=(10, 6))
plt.plot(omega, H_db)
plt.title('Magnitude Response of Lowpass Filter')
plt.xlabel('Frequency [radians/sample]')
plt.ylabel('Magnitude [dB]')
plt.grid(True)
plt.show()

# 绘制相位响应
plt.plot(omega, H_phase)
plt.title('Phase Response of Lowpass Filter')
plt.xlabel('Frequency [radians/sample]')
plt.ylabel('Phase [radians]')
plt.grid(True)
plt.show()

```
请根据使用最小阶低通滤波器程序，完成Python语言firpmord函数的编写，便于类似应用调用。

### LLM 生成 ： Python firpmord 函数

这是firpmord函数定义
def firpmord(f, a, dev, fs):
    """
    Estimate the minimum order for a Parks-McClellan (firpm) FIR filter.
    
    Parameters:
        f (array_like): Cutoff frequencies.
        a (array_like): Desired amplitudes.
        dev (array_like): Maximum passband and minimum stopband ripple.
        fs (float): Sampling frequency.
    
    Returns:
        n (int): Minimum filter order.
        fo (ndarray): Optimal frequencies.
        ao (ndarray): Optimal amplitudes.
        w (ndarray): Optimal weights.
    """
    assert len(f) == len(a) == len(dev) == 2, "Input arrays must have length 2."
    
    rp, rs = dev
    delta_p = (10 ** (rp / 20) - 1) / (10 ** (rp / 20) + 1)
    delta_s = 10 ** (-rs / 20)
    
    def objective(n):
        delta = np.array([delta_p, delta_s])
        m = (n - 1) // 2
        k = np.arange(1, m + 1)
        h = np.sinc(f[1] * k / fs) - np.sinc(f[0] * k / fs)
        a_m = 2 * np.sum(np.cos(2 * np.pi * f[1] * k / fs) * h)
        d_m = 2 * np.sum(np.cos(2 * np.pi * f[1] * k / fs) * np.sin(np.pi * k / m) / (np.pi * k))
        d_m += np.sum(np.cos(2 * np.pi * f[0] * k / fs) * np.sin(np.pi * k / m) / (np.pi * k))
        
        # Modify alpha calculation to ensure shape compatibility
        alpha = np.sum(np.abs(1 + (-1) ** k * a_m))
        delta_a = alpha - 1
        delta_d = (1 - d_m) / 2
        return delta @ np.array([delta_a, delta_d])
    
    res = minimize_scalar(objective, method='bounded', bounds=(1, 1000))
    n = int(np.ceil(res.x))
    
    m = (n - 1) // 2
    fo = np.array([0, f[0] / fs, f[1] / fs, 0.5])
    ao = np.array([a[0], a[0], a[1], a[1]])
    w = np.array([1, delta_p / delta_s, 1])
    
    return n, fo, ao, w


## Prompt 2 ： 应用生成的 Python firpmord 函数做MATLAB示例对应

这是初始信号参数rp、rs、fs、f、a、dev
```
# 初始信号参数
rp = 3            # 通带波纹（dB）
rs = 40           # 阻带波纹（dB）
fs = 2000         # 采样频率（Hz）
f = [500, 600]    # 截止频率（Hz）
a = [1, 0]        # 期望幅度

# 使用 firpmord 函数估算最小阶数和最优参数
dev = [(10**(rp/20)-1)/(10**(rp/20)+1), 10**(-rs/20)]
```

这是firpmord函数定义
def firpmord(f, a, dev, fs):
    """
    Estimate the minimum order for a Parks-McClellan (firpm) FIR filter.
    
    Parameters:
        f (array_like): Cutoff frequencies.
        a (array_like): Desired amplitudes.
        dev (array_like): Maximum passband and minimum stopband ripple.
        fs (float): Sampling frequency.
    
    Returns:
        n (int): Minimum filter order.
        fo (ndarray): Optimal frequencies.
        ao (ndarray): Optimal amplitudes.
        w (ndarray): Optimal weights.
    """
    assert len(f) == len(a) == len(dev) == 2, "Input arrays must have length 2."
    
    rp, rs = dev
    delta_p = (10 ** (rp / 20) - 1) / (10 ** (rp / 20) + 1)
    delta_s = 10 ** (-rs / 20)
    
    def objective(n):
        delta = np.array([delta_p, delta_s])
        m = (n - 1) // 2
        k = np.arange(1, m + 1)
        h = np.sinc(f[1] * k / fs) - np.sinc(f[0] * k / fs)
        a_m = 2 * np.sum(np.cos(2 * np.pi * f[1] * k / fs) * h)
        d_m = 2 * np.sum(np.cos(2 * np.pi * f[1] * k / fs) * np.sin(np.pi * k / m) / (np.pi * k))
        d_m += np.sum(np.cos(2 * np.pi * f[0] * k / fs) * np.sin(np.pi * k / m) / (np.pi * k))
        
        # Modify alpha calculation to ensure shape compatibility
        alpha = np.sum(np.abs(1 + (-1) ** k * a_m))
        delta_a = alpha - 1
        delta_d = (1 - d_m) / 2
        return delta @ np.array([delta_a, delta_d])
    
    res = minimize_scalar(objective, method='bounded', bounds=(1, 1000))
    n = int(np.ceil(res.x))
    
    m = (n - 1) // 2
    fo = np.array([0, f[0] / fs, f[1] / fs, 0.5])
    ao = np.array([a[0], a[0], a[1], a[1]])
    w = np.array([1, delta_p / delta_s, 1])
    
    return n, fo, ao, w


请给出应用firpmord函数对初始信号参数rp、rs、fs、f、a、dev设计最小阶低通滤波器程序，并绘制图形，给出Python程序。

### LLM 生成 ： 
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize_scalar

def firpmord(f, a, dev, fs):
    """
    Estimate the minimum order for a Parks-McClellan (firpm) FIR filter.
    
    Parameters:
        f (array_like): Cutoff frequencies.
        a (array_like): Desired amplitudes.
        dev (array_like): Maximum passband and minimum stopband ripple.
        fs (float): Sampling frequency.
    
    Returns:
        n (int): Minimum filter order.
        fo (ndarray): Optimal frequencies.
        ao (ndarray): Optimal amplitudes.
        w (ndarray): Optimal weights.
    """
    assert len(f) == len(a) == len(dev) == 2, "Input arrays must have length 2."
    
    rp, rs = dev
    delta_p = (10 ** (rp / 20) - 1) / (10 ** (rp / 20) + 1)
    delta_s = 10 ** (-rs / 20)
    
    def objective(n):
        delta = np.array([delta_p, delta_s])
        m = (n - 1) // 2
        k = np.arange(1, m + 1)
        h = np.sinc(f[1] * k / fs) - np.sinc(f[0] * k / fs)
        a_m = 2 * np.sum(np.cos(2 * np.pi * f[1] * k / fs) * h)
        d_m = 2 * np.sum(np.cos(2 * np.pi * f[1] * k / fs) * np.sin(np.pi * k / m) / (np.pi * k))
        d_m += np.sum(np.cos(2 * np.pi * f[0] * k / fs) * np.sin(np.pi * k / m) / (np.pi * k))
        
        # Modify alpha calculation to ensure shape compatibility
        alpha = np.sum(np.abs(1 + (-1) ** k * a_m))
        delta_a = alpha - 1
        delta_d = (1 - d_m) / 2
        return delta @ np.array([delta_a, delta_d])
    
    res = minimize_scalar(objective, method='bounded', bounds=(1, 1000))
    n = int(np.ceil(res.x))
    
    m = (n - 1) // 2
    fo = np.array([0, f[0] / fs, f[1] / fs, 0.5])
    ao = np.array([a[0], a[0], a[1], a[1]])
    w = np.array([1, delta_p / delta_s, 1])
    
    return n, fo, ao, w

def firpm(N, f, a):
    """
    Python implementation of firpm function similar to MATLAB's firpm.
    
    Parameters:
        N (int): Filter order.
        f (array_like): Array of frequency points where the amplitude values are specified.
                        The frequencies are normalized to the Nyquist frequency, so they must be between 0 and 1.
        a (array_like): Array of amplitude values corresponding to the frequency points in f.
    
    Returns:
        b (ndarray): Coefficients of the filter.
    """
    # Ensure that f and a have the same length
    if len(f) != len(a):
        raise ValueError("Lengths of frequency and amplitude arrays must be the same.")
    
    # Design the filter using Parks-McClellan algorithm
    b = signal.firwin2(N, f, a)
    
    return b

# 初始信号参数
rp = 3            # 通带波纹（dB）
rs = 40           # 阻带波纹（dB）
fs = 2000         # 采样频率（Hz）
f = [500, 600]    # 截止频率（Hz）
a = [1, 0]        # 期望幅度

# 使用 firpmord 函数估算最小阶数和最优参数
dev = [(10**(rp/20)-1)/(10**(rp/20)+1), 10**(-rs/20)]
n, fo, ao, w = firpmord(f, a, dev, fs)
# Define the sampling frequency and the Nyquist frequency
nyquist = fs / 2

# Define the cutoff frequencies normalized by the Nyquist frequency
f_normalized = [0, f[0] / nyquist, f[1] / nyquist, 1]

# Define the amplitudes at each of these frequencies
a_normalized = [1, 1, 0, 0]  # Passband from 0 to f[0], stopband from f[1] to Nyquist

# Design the filter using the corrected frequency and amplitude arrays
b = firpm(n, f_normalized, a_normalized)

# Frequency and complex response
freq, response = signal.freqz(b, worN=2000, fs=fs)

# Calculate phase response
phase = np.unwrap(np.angle(response))

# Plot magnitude and phase responses
plt.figure(figsize=(12, 10))

# Magnitude response
plt.subplot(2, 1, 1)
plt.plot(freq, 20 * np.log10(np.abs(response)), label='Magnitude Response')
plt.title('Frequency Response of the Designed FIR Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.grid(True)
plt.legend()

# Phase response
plt.subplot(2, 1, 2)
plt.plot(freq, np.degrees(phase), label='Phase Response')
plt.title('Phase Response of the Designed FIR Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```

## Prompt 3 ： 应用生成的 Python firpmord.py 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：

**Matlab代码**
```
rp = 3;           % Passband ripple in dB 
rs = 40;          % Stopband ripple in dB
fs = 2000;        % Sampling frequency
f = [500 600];    % Cutoff frequencies
a = [1 0];        % Desired amplitudes

dev = [(10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)]; 
[n,fo,ao,w] = firpmord(f,a,dev,fs);
b = firpm(n,fo,ao,w);
freqz(b,1,1024,fs)
title('Lowpass Filter Designed to Specifications')
```

这是Python firpmord.py
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize_scalar

def firpmord(f, a, dev, fs):
    """
    Estimate the minimum order for a Parks-McClellan (firpm) FIR filter.
    
    Parameters:
        f (array_like): Cutoff frequencies.
        a (array_like): Desired amplitudes.
        dev (array_like): Maximum passband and minimum stopband ripple.
        fs (float): Sampling frequency.
    
    Returns:
        n (int): Minimum filter order.
        fo (ndarray): Optimal frequencies.
        ao (ndarray): Optimal amplitudes.
        w (ndarray): Optimal weights.
    """
    assert len(f) == len(a) == len(dev) == 2, "Input arrays must have length 2."
    
    rp, rs = dev
    delta_p = (10 ** (rp / 20) - 1) / (10 ** (rp / 20) + 1)
    delta_s = 10 ** (-rs / 20)
    
    def objective(n):
        delta = np.array([delta_p, delta_s])
        m = (n - 1) // 2
        k = np.arange(1, m + 1)
        h = np.sinc(f[1] * k / fs) - np.sinc(f[0] * k / fs)
        a_m = 2 * np.sum(np.cos(2 * np.pi * f[1] * k / fs) * h)
        d_m = 2 * np.sum(np.cos(2 * np.pi * f[1] * k / fs) * np.sin(np.pi * k / m) / (np.pi * k))
        d_m += np.sum(np.cos(2 * np.pi * f[0] * k / fs) * np.sin(np.pi * k / m) / (np.pi * k))
        
        # Modify alpha calculation to ensure shape compatibility
        alpha = np.sum(np.abs(1 + (-1) ** k * a_m))
        delta_a = alpha - 1
        delta_d = (1 - d_m) / 2
        return delta @ np.array([delta_a, delta_d])
    
    res = minimize_scalar(objective, method='bounded', bounds=(1, 1000))
    n = int(np.ceil(res.x))
    
    m = (n - 1) // 2
    fo = np.array([0, f[0] / fs, f[1] / fs, 0.5])
    ao = np.array([a[0], a[0], a[1], a[1]])
    w = np.array([1, delta_p / delta_s, 1])
    
    return n, fo, ao, w
```


### LLM 生成 ： 

下面这是调用程序
```
from firpm import firpm
from firpmord import firpmord
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize_scalar

# 使用firpmord和firpm函数设计滤波器
f = [1500, 2000]  # Cutoff frequencies (Hz)
a = [1, 0]        # Desired amplitudes
dev = [0.01, 0.1] # Passband and stopband ripple specifications
fs = 8000         # Sampling frequency (Hz)

# Estimate filter order and obtain filter parameters
n, fo, ao, w = firpmord(f, a, dev, fs)

# Design the filter using firpm
b = firpm(n, fo, ao)

# Frequency and complex response
freq, response = signal.freqz(b, worN=8000, fs=fs)

# Calculate phase response
phase = np.unwrap(np.angle(response))

# Plot magnitude and phase responses
plt.figure(figsize=(12, 10))

# Magnitude response
plt.subplot(2, 1, 1)
plt.plot(freq, 20 * np.log10(np.abs(response)), label='Magnitude Response')
plt.title('Frequency Response of the Designed FIR Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.grid(True)
plt.legend()

# Phase response
plt.subplot(2, 1, 2)
plt.plot(freq, np.degrees(phase), label='Phase Response')
plt.title('Phase Response of the Designed FIR Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```






