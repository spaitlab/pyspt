# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：gaussdesign 

函数来源：[MATLAB gaussdesign](https://ww2.mathworks.cn/help/signal/ref/gaussdesign.html)

### 语法

h = gaussdesign(bt,span,sps)


### 说明

h = gaussdesign（bt，span，sps）设计了一个低通FIR高斯脉冲整形滤波器，并返回滤波器系数的向量。筛选器被截断为符号，并且每个符号周期都包含样本。筛选器的顺序必须为偶数。

### 输入参数

bt — 3-dB 带宽-符号时间乘积
正实数标量
3-dB 单边带宽和符号时间的乘积，指定为正实数标量。3-dB 单边带宽以赫兹为单位，符号时间以秒为单位。较小的值会产生较大的脉冲宽度。

span — 符号数量
3（默认） | 正整数标量
符号数量，指定为正整数标量。

sps — 每个符号的样本数
2（默认） | 正整数标量
每个符号周期内的样本数（过采样因子），指定为正整数标量。

### 输出参量

h— FIR 滤波器系数
向量
返回的高斯脉冲整形滤波器的 FIR 系数 作为行向量。系数被归一化，使名义 通带增益始终为 1。

数据类型：double


## Python函数描述：gaussdesign

函数来源：自定义

### 高斯滤波器函数定义：

def gaussdesign(bt, span=3, sps=2):
    """
    Design a Gaussian FIR Pulse-Shaping Filter.

    Parameters:
        bt (float): 3 dB bandwidth-symbol time product.
        span (int, optional): Total length of the filter in symbols. Default is 3.
        sps (int, optional): Number of samples per symbol. Default is 2.

    Returns:
        h (ndarray): Coefficients of the Gaussian filter.
    """
    # Check if filter order is even
    sps_span = sps * span
    if sps_span % 2 != 0:
        raise ValueError("Filter order must be even.")

    # Calculate filter length
    filt_len = sps_span + 1

    # Convert to t in which to compute the filter coefficients
    t = np.linspace(-span / 2, span / 2, filt_len)

    # Compute alpha
    alpha = np.sqrt(np.log(2) / 2) / bt

    # Compute filter coefficients
    h = (np.sqrt(np.pi) / alpha) * np.exp(-(t * np.pi / alpha) ** 2)

    # Normalize coefficients
    h /= np.sum(h)

    return h


### 参数

- `bt（float）`：3 dB 带宽-符号时间乘积。
- `span（int, optional）`：滤波器的总长度，以符号为单位。默认为 3。
- `sps（int, optional）`：每个符号的样本数。默认为 2。

### 返回值

- `h (ndarray)`: 高斯滤波器的系数。

### 函数工作原理

该函数设计高斯 FIR 脉冲整形滤波器。它使用高斯函数的形式来设计滤波器，以实现对数字信号的脉冲整形。

### 使用场景

1.适用于数字通信系统中的脉冲整形：在数字通信系统中，经常需要使用脉冲整形滤波器来塑造发送信号的脉冲形状，以满足频谱和时域限制。
2.高斯频率调制：在高斯频率调制中，需要使用高斯脉冲整形滤波器来塑造调制信号的频谱。

### 改进建议

- 参数灵活性提升：考虑增加更多参数选项，如滤波器类型（低通、带通等）、滤波器阶数等，以增加函数的灵活性和适用性。
- 文档和注释完善：提供更详细的函数文档和注释，解释函数的输入参数和输出结果的含义，以帮助用户更好地理解和使用该函数。
- 性能优化：优化算法实现，以提高函数的计算效率和性能。可能的优化包括减少计算复杂度或者利用并行计算来加速计算过程。



## Prompt 1 ： 生成 Python gaussdesign 函数

参考下面MATLAB代码的gaussdesign函数
```
bt = 0.3;
span = 4;
sps = 8;
h = gaussdesign(bt,span,sps);
impz(h)

```

和我们采用Python语言实现的用于 GSM GMSK 数字蜂窝通信系统的高斯滤波器，
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian

# Define parameters
bt = 0.3  # 3-dB bandwidth-symbol time product
span = 4  # Total length of the filter in symbols
sps = 8   # Number of samples per symbol

# Calculate filter length
filt_len = span * sps

# Generate Gaussian pulse
t = np.arange(-filt_len // 2, filt_len // 2)
h = gaussian(filt_len, std=filt_len / (2 * np.pi * bt))

# Normalize coefficients
h /= np.sum(h)

# Plot impulse response
plt.stem(h)
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Impulse Response of Gaussian Filter')
plt.grid(True)
plt.show()

```
请根据使用GSM GMSK 数字蜂窝通信系统的高斯滤波器程序，完成Python语言gaussdesign函数的编写，便于类似应用调用。

### LLM 生成 ： Python gaussdesign 函数

这是gaussdesign函数定义
def gaussdesign(bt, span=3, sps=2):
    """
    Design a Gaussian FIR Pulse-Shaping Filter.

    Parameters:
        bt (float): 3 dB bandwidth-symbol time product.
        span (int, optional): Total length of the filter in symbols. Default is 3.
        sps (int, optional): Number of samples per symbol. Default is 2.

    Returns:
        h (ndarray): Coefficients of the Gaussian filter.
    """
    # Check if filter order is even
    sps_span = sps * span
    if sps_span % 2 != 0:
        raise ValueError("Filter order must be even.")

    # Calculate filter length
    filt_len = sps_span + 1

    # Convert to t in which to compute the filter coefficients
    t = np.linspace(-span / 2, span / 2, filt_len)

    # Compute alpha
    alpha = np.sqrt(np.log(2) / 2) / bt

    # Compute filter coefficients
    h = (np.sqrt(np.pi) / alpha) * np.exp(-(t * np.pi / alpha) ** 2)

    # Normalize coefficients
    h /= np.sum(h)

    return h


## Prompt 2 ： 应用生成的 Python gaussdesign 函数做MATLAB示例对应

这是参数bt,symbol_length,samples_per_symbol
```
import numpy as np
import matplotlib.pyplot as plt

# Parameters
bt = 0.3  # 3-dB bandwidth
symbol_length = 4  # Number of symbols
samples_per_symbol = 8  # Number of samples per symbol
```

这是gaussdesign函数定义
def gaussdesign(bt, span=3, sps=2):
    """
    Design a Gaussian FIR Pulse-Shaping Filter.

    Parameters:
        bt (float): 3 dB bandwidth-symbol time product.
        span (int, optional): Total length of the filter in symbols. Default is 3.
        sps (int, optional): Number of samples per symbol. Default is 2.

    Returns:
        h (ndarray): Coefficients of the Gaussian filter.
    """
    # Check if filter order is even
    sps_span = sps * span
    if sps_span % 2 != 0:
        raise ValueError("Filter order must be even.")

    # Calculate filter length
    filt_len = sps_span + 1

    # Convert to t in which to compute the filter coefficients
    t = np.linspace(-span / 2, span / 2, filt_len)

    # Compute alpha
    alpha = np.sqrt(np.log(2) / 2) / bt

    # Compute filter coefficients
    h = (np.sqrt(np.pi) / alpha) * np.exp(-(t * np.pi / alpha) ** 2)

    # Normalize coefficients
    h /= np.sum(h)

    return h

请给出应用gaussdesign函数对参数bt,symbol_length,samples_per_symbol设计GSM GMSK 数字蜂窝通信系统的高斯滤波器，并绘制图形，给出Python程序。

### LLM 生成 ： 
```
import numpy as np
import matplotlib.pyplot as plt
def gaussdesign(bt, span=3, sps=2):
    """
    Design a Gaussian FIR Pulse-Shaping Filter.

    Parameters:
        bt (float): 3 dB bandwidth-symbol time product.
        span (int, optional): Total length of the filter in symbols. Default is 3.
        sps (int, optional): Number of samples per symbol. Default is 2.

    Returns:
        h (ndarray): Coefficients of the Gaussian filter.
    """
    # Check if filter order is even
    sps_span = sps * span
    if sps_span % 2 != 0:
        raise ValueError("Filter order must be even.")

    # Calculate filter length
    filt_len = sps_span + 1

    # Convert to t in which to compute the filter coefficients
    t = np.linspace(-span / 2, span / 2, filt_len)

    # Compute alpha
    alpha = np.sqrt(np.log(2) / 2) / bt

    # Compute filter coefficients
    h = (np.sqrt(np.pi) / alpha) * np.exp(-(t * np.pi / alpha) ** 2)

    # Normalize coefficients
    h /= np.sum(h)

    return h
def impz(b, a=[1], N=20, Fs=1):
    """
    Impulse response plot for discrete-time systems.

    Parameters:
        b (array_like): Numerator coefficients of the system.
        a (array_like, optional): Denominator coefficients of the system. Default is [1].
        N (int, optional): Number of points to plot. Default is 20.
        Fs (float, optional): Sampling frequency. Default is 1.

    Returns:
        None (plots the impulse response).
    """
    # Compute impulse response
    impulse = np.zeros(N)
    impulse[0] = 1
    response = np.convolve(b, impulse)

    # Plot impulse response
    n = np.arange(0, len(response)) / Fs
    plt.stem(n, response, basefmt=' ')  # Remove the red baseline
    plt.xlabel('(n) samples')
    plt.ylabel('Amplitude')
    plt.title('Impulse Response')
    plt.xlim([0, 30])  # Limit x-axis from 0 to 30
    plt.grid(True)
    plt.show()

# Parameters
bt = 0.3  # 3-dB bandwidth
symbol_length = 4  # Number of symbols
samples_per_symbol = 8  # Number of samples per symbol

# Generate Gaussian filter coefficients
h = gaussdesign(bt, symbol_length, samples_per_symbol)

# Plot the impulse response
impz(h)
```

## Prompt 3 ： 应用生成的 Python gaussdesign.py 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：

**Matlab代码**
```
bt = 0.3;
span = 4;
sps = 8;
h = gaussdesign(bt,span,sps);
impz(h)
```

这是Python gaussdesign.py
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import freqz


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
```


### LLM 生成 ： 

下面这是调用程序
```
from gaussdesign import gaussdesign
from impz import impz
import numpy as np
import matplotlib.pyplot as plt

# Parameters
bt = 0.3  # 3-dB bandwidth
symbol_length = 4  # Number of symbols
samples_per_symbol = 8  # Number of samples per symbol

# Generate Gaussian filter coefficients
h = gaussdesign(bt, symbol_length, samples_per_symbol)

# Plot the impulse response
impz(h)
```





