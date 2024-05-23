# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：designfilt 

函数来源：[MATLAB designfilt]([Design digital filters - MATLAB designfilt - MathWorks 中国](https://ww2.mathworks.cn/help/signal/ref/designfilt.html))

### 语法

```
d = designfilt(resp,Name,Value)
designfilt(d)
```

### 说明

```
d = designfilt（resp，Name，Value）` 设计一个 [`digitalFilter`](https://ww2.mathworks.cn/help/signal/ref/digitalfilter.html) 对象 [`d`](https://ww2.mathworks.cn/help/signal/ref/designfilt.html#bt61003-d-dup1)，其响应类型[`为 resp`](https://ww2.mathworks.cn/help/signal/ref/designfilt.html#bt61003-resp)。例如[`“lowpassfir”`](https://ww2.mathworks.cn/help/signal/ref/designfilt.html#mw_685db941-ffa4-41ff-8225-a4fec9532a22)和[`“bandstopiir”`](https://ww2.mathworks.cn/help/signal/ref/designfilt.html#mw_fcc4648b-26a0-42ff-8a33-0d4e536d3b16)。 使用一组[名称-值参数](https://ww2.mathworks.cn/help/signal/ref/designfilt.html#bt612s9-1)进一步指定筛选器。允许的规范集取决于并由以下组合组成：`resp``resp
```

- *频率约束*对应于 滤波器表现出所需行为的频率。例子 包括[`“PassbandFrequency”`](https://ww2.mathworks.cn/help/signal/ref/designfilt.html#bt61003-PassbandFrequency)和[`“CutoffFrequency”`](https://ww2.mathworks.cn/help/signal/ref/designfilt.html#bt61003-CutoffFrequency)。 您必须始终指定频率约束。
- *幅度约束*描述了特定频率下的滤波器行为 范围。示例包括[`“PassbandRipple”`](https://ww2.mathworks.cn/help/signal/ref/designfilt.html#bt61003-PassbandRipple)和[`“StopbandAttenuation”`](https://ww2.mathworks.cn/help/signal/ref/designfilt.html#bt61003-StopbandAttenuation)。 提供未指定的幅度约束的默认值。在 任意量级设计，您必须始终指定 所需的振幅。`designfilt`
- *筛选顺序*：某些设计方法允许您指定顺序。 其他人则生产最小阶设计。也就是说，它们生成 满足指定约束条件的最小筛选器。
- *设计方法是*用于设计滤波器的算法。例子 包括约束最小二乘法 （） 和 Kaiser 窗口化 （）。对于某些规格集， 有多种设计方法可供选择。在其他 情况下，您只能使用一种方法来满足所需的要求 规格。`'cls'``'kaiserwin'`
- *设计方法选项*是特定于给定设计方法的参数。 示例包括方法的[`“Window”`](https://ww2.mathworks.cn/help/signal/ref/designfilt.html#bt61003-Window)和优化[`的“Weights”`](https://ww2.mathworks.cn/help/signal/ref/designfilt.html#bt61003-Weights) 任意幅度等纹波设计。 为未指定的设计选项提供默认值。`'window'``designfilt`
- *采样率*是滤波器工作频率。 默认采样率为 2 Hz。 使用此值等同于使用规范化 频率。`designfilt`
- 使用 [`filter`](https://ww2.mathworks.cn/help/matlab/ref/filter.html) 函数 要过滤的形式 带有 .对于 IIR 滤波器，该函数使用直接形式的 II 实现。您还可以使用 [`filtfilt`](https://ww2.mathworks.cn/help/signal/ref/filtfilt.html) 和 [`fftfilt`](https://ww2.mathworks.cn/help/signal/ref/fftfilt.html) 函数 与对象。`dataOut = filter(d,dataIn)``dataIn``digitalFilter``d``filter``digitalFilter`
- 使用[过滤器分析器](https://ww2.mathworks.cn/help/signal/ref/filteranalyzer-app.html)可视化过滤器。`d`
- 类型以获取系数 的 .对于IIR滤波器，系数为： 表示为二阶截面。`d.Coefficients``digitalFilter``d`
- 请参阅 [`digitalFilter`](https://ww2.mathworks.cn/help/signal/ref/digitalfilter.html) 了解 可用于对象的筛选和分析函数的列表。`digitalFilter`

### 输入参数

resp— 滤波器响应和类型
'lowpassfir' |“Lowpassiir” |“高帕斯菲尔” |“Highpassiir” |“Bandpassfir”（班帕斯菲尔酒店） |“Bandpassiir”
筛选器响应和类型，指定为字符向量或字符串标量。

'lowpassfir'— FIR低通滤波器
响应类型
选择此选项可设计有限脉冲响应 （FIR） 低通滤波器。

'lowpassiir'— IIR低通滤波器
响应类型
选择此选项可设计无限脉冲响应 （IIR）低通滤波器。此示例使用第一个规范从桌面上设置。

'highpassfir'— FIR高通滤波器
响应类型
选择此选项可设计有限脉冲响应 （FIR） 高通滤波器。此示例使用第一个规范集。

'highpassiir'— IIR高通滤波器
响应类型
选择此选项可设计无限脉冲响应 （IIR）高通滤波器。此示例使用第一个规范。

'bandpassfir'— FIR带通滤波器
响应类型
选择此选项可设计有限脉冲响应 （FIR） 带通滤波器。此示例使用第四个规范集。

'bandpassiir'— IIR带通滤波器
响应类型
选择此选项可设计无限脉冲响应 （IIR）带通滤波器。此示例使用第一个规范。

'bandstopfir'— FIR带阻滤波器
响应类型
选择此选项可设计有限脉冲响应 （FIR） 带阻滤波器。此示例使用第四个规范集。

'bandstopiir'— IIR带阻滤波器
响应类型
选择此选项可设计无限脉冲响应 （IIR） 带阻滤波器。此示例使用第一个规范。

'differentiatorfir'— FIR微分滤波器
响应类型
选择此选项可设计有限脉冲响应 （FIR） 微分器过滤器。此示例使用第二个规范。

'hilbertfir'— FIR 希尔伯特变压器滤波器
响应类型
选择此选项可设计有限脉冲响应 （FIR） 希尔伯特变压器滤波器。此示例使用规范。

'arbmagfir'— 任意幅度响应
响应类型的FIR滤波器
选择此选项可设计有限脉冲响应 （FIR） 任意幅度响应的滤波器。此示例使用表中的第二个规格集。

d— 数字滤波器
digitalFilter 对象
数字筛选器，指定为 digitalFilter 对象生成者。使用此输入可更改现有 .designfiltdigitalFilter。

### 输出参量

d— 数字滤波器
digitalFilter 对象
数字筛选器，作为 digitalFilter 对象返回。

## Python函数描述：designfilt

函数来源：自定义

### 函数定义：

def design_bandpass_fir(order, cutoff_low, cutoff_high, fs):
    ...
    return coeffs

    # designfilt.py
    import numpy as np
    from scipy.signal import firwin, freqz
    import matplotlib.pyplot as plt
    
    def design_bandpass_fir(order, cutoff_low, cutoff_high, fs):
        """
        Designs a bandpass FIR filter using the window method.
        
        Parameters:
        order (int): The order of the filter.
        cutoff_low (float): The lower cutoff frequency (in Hz).
        cutoff_high (float): The higher cutoff frequency (in Hz).
        fs (int): The sampling frequency (in Hz).
        
        Returns:
        ndarray: The coefficients of the FIR filter.
        """
        nyq = fs / 2.0
        low = cutoff_low / nyq
        high = cutoff_high / nyq
        coeffs = firwin(order + 1, [low, high], pass_zero=False)
        return coeffs
    
    def visualize_filter_response(coeffs, fs):
        """
        Visualizes the frequency and phase response of the FIR filter.
        
        Parameters:
        coeffs (ndarray): The coefficients of the FIR filter.
        fs (int): The sampling frequency (in Hz).
        """
        w, h = freqz(coeffs, worN=8000)
        w = w * fs / (2 * np.pi)  # Convert from rad/sample to Hz
    
        plt.figure(figsize=(12, 6))
    
        # Magnitude response
        plt.subplot(2, 1, 1)
        plt.plot(w, 20 * np.log10(np.abs(h)), 'b')
        plt.title('Magnitude Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
    
        # Phase response
        plt.subplot(2, 1, 2)
        plt.plot(w, np.unwrap(np.angle(h)), 'b')
        plt.title('Phase Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (degrees)')
        plt.grid(True)
    
        plt.tight_layout()
        plt.show()

这段代码定义了一个名为 `design_bandpass_fir` 的函数，其目的是通过窗函数法设计一个带通 FIR 滤波器。窗函数法是一种用于生成滤波器系数的数学工具，它通过将理想的频率响应与一个窗函数相乘来实现。这种方法可以有效地控制滤波器的通带和阻带特性，从而精确地定义滤波器在特定频率范围内的行为。

这段代码还定义了一个名为 `visualize_filter_response` 的函数，其目的是可视化 FIR 滤波器的频率响应和相位响应。频率响应的可视化是一种用于展示滤波器效果的图形表示方法，它可以用于展示滤波器在各个频率上的增益和相位变化。这种方法特别适用于分析和验证滤波器设计的性能，确保滤波器符合预定的规格要求。

### 参数
- **order** (`int`): 滤波器的阶数，决定了滤波器的复杂度和特性。
- **cutoff_low** (`float`): 滤波器的低频截止频率（单位：赫兹）。
- **cutoff_high** (`float`): 滤波器的高频截止频率（单位：赫兹）。
- **fs** (`int`): 采样频率（单位：赫兹）。
- **coeffs** (`ndarray`): FIR滤波器的系数。
- **fs** (`int`): 采样频率（单位：赫兹）。

### 返回值
- **ndarray**: FIR滤波器的系数数组。

### 注意事项
- 确保 `cutoff_low` 和 `cutoff_high` 在 0 和 Nyquist 频率之间。
- 函数依赖于matplotlib库进行图形显示，确保已正确安装此库。

### 函数工作原理
1. 该函数利用窗函数方法设计一个带通FIR滤波器。通过firwin函数创建一个具有指定低频和高频截止频率的带通滤波器，其中使用了窗函数来控制滤波器的过渡带和旁瓣衰减。
2. 该函数显示FIR滤波器的频率响应和相位响应。首先计算滤波器的频率响应，然后分别在两个子图中绘制幅度响应和相位响应。

### 使用场景
适用于需要在特定频率范围内过滤信号的应用，如音频信号处理、生物信号处理等。用于分析和验证滤波器设计的正确性和效果，特别是在调整设计参数后。

### 改进建议
- 可以考虑引入不同类型的窗函数以满足不同的设计要求。



## Prompt 1 ： 生成 Python designfilt 函数

参考下面MATLAB代码的designfilt函数
```
bpfilt = designfilt('bandpassfir', ...
    'FilterOrder',20,'CutoffFrequency1',400, ...
    'CutoffFrequency2',450,'SampleRate',1500);
freqz(bpfilt.Coefficients,1,[],1500)
```

请根据使用Python语言完成designfilt函数的编写，便于类似应用调用。

### LLM 生成 ： Python designfilt 函数

这是函数定义
def design_bandpass_fir(order, cutoff_low, cutoff_high, fs):
    ...
    return coeffs

    # designfilt.py
    import numpy as np
    from scipy.signal import firwin, freqz
    import matplotlib.pyplot as plt
    
    def design_bandpass_fir(order, cutoff_low, cutoff_high, fs):
        """
        Designs a bandpass FIR filter using the window method.
        
        Parameters:
        order (int): The order of the filter.
        cutoff_low (float): The lower cutoff frequency (in Hz).
        cutoff_high (float): The higher cutoff frequency (in Hz).
        fs (int): The sampling frequency (in Hz).
        
        Returns:
        ndarray: The coefficients of the FIR filter.
        """
        nyq = fs / 2.0
        low = cutoff_low / nyq
        high = cutoff_high / nyq
        coeffs = firwin(order + 1, [low, high], pass_zero=False)
        return coeffs
    
    def visualize_filter_response(coeffs, fs):
        """
        Visualizes the frequency and phase response of the FIR filter.
        
        Parameters:
        coeffs (ndarray): The coefficients of the FIR filter.
        fs (int): The sampling frequency (in Hz).
        """
        w, h = freqz(coeffs, worN=8000)
        w = w * fs / (2 * np.pi)  # Convert from rad/sample to Hz
    
        plt.figure(figsize=(12, 6))
    
        # Magnitude response
        plt.subplot(2, 1, 1)
        plt.plot(w, 20 * np.log10(np.abs(h)), 'b')
        plt.title('Magnitude Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
    
        # Phase response
        plt.subplot(2, 1, 2)
        plt.plot(w, np.unwrap(np.angle(h)), 'b')
        plt.title('Phase Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (degrees)')
        plt.grid(True)
    
        plt.tight_layout()
        plt.show()

## Prompt 2 ： 应用生成的 Python designfilt.py 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：

注意：
(1)Matlab和Python对函数的名称不一样，参数定义不一样，需要换算。
(2)线条采用不同颜色方便区分。

**Matlab代码**

```
bpfilt = designfilt('bandpassfir', ...
    'FilterOrder',20,'CutoffFrequency1',400, ...
    'CutoffFrequency2',450,'SampleRate',1500);
freqz(bpfilt.Coefficients,1,[],1500)
```

这是Python designfilt.py
```
# designfilt.py
import numpy as np
from scipy.signal import firwin, freqz
import matplotlib.pyplot as plt

def design_bandpass_fir(order, cutoff_low, cutoff_high, fs):
    """
    Designs a bandpass FIR filter using the window method.
    
    Parameters:
    order (int): The order of the filter.
    cutoff_low (float): The lower cutoff frequency (in Hz).
    cutoff_high (float): The higher cutoff frequency (in Hz).
    fs (int): The sampling frequency (in Hz).
    
    Returns:
    ndarray: The coefficients of the FIR filter.
    """
    nyq = fs / 2.0
    low = cutoff_low / nyq
    high = cutoff_high / nyq
    coeffs = firwin(order + 1, [low, high], pass_zero=False)
    return coeffs

def visualize_filter_response(coeffs, fs):
    """
    Visualizes the frequency and phase response of the FIR filter.
    
    Parameters:
    coeffs (ndarray): The coefficients of the FIR filter.
    fs (int): The sampling frequency (in Hz).
    """
    w, h = freqz(coeffs, worN=8000)
    w = w * fs / (2 * np.pi)  # Convert from rad/sample to Hz

    plt.figure(figsize=(12, 6))

    # Magnitude response
    plt.subplot(2, 1, 1)
    plt.plot(w, 20 * np.log10(np.abs(h)), 'b')
    plt.title('Magnitude Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)

    # Phase response
    plt.subplot(2, 1, 2)
    plt.plot(w, np.unwrap(np.angle(h)), 'b')
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
```


### LLM 生成 ： 

下面这是调用程序
```
# other_python_environment.py

from designfilt import design_bandpass_fir, visualize_filter_response

# Design the filter with given specifications
filter_coeffs = design_bandpass_fir(order=20, cutoff_low=400, cutoff_high=450, fs=1500)

# Visualize the frequency response
visualize_filter_response(filter_coeffs, fs=1500)
```




