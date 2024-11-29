# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：cfirpm

函数来源：[MATLAB cfirpm]([Complex and nonlinear-phase equiripple FIR filter design - MATLAB cfirpm - MathWorks 中国](https://ww2.mathworks.cn/help/signal/ref/cfirpm.html))

### 语法

```
b = cfirpm(n,f,fresp)
b = cfirpm(n,f,fresp,w)
b = cfirpm(n,f,a)
b = cfirpm(n,f,a,w)
b = cfirpm(___,sym)
b = cfirpm(___,debug)
b = cfirpm(___,lgrid)
b = cfirpm(___,'skip_stage2')
[b,delta] = cfirpm(___)
[b,delta,opt] = cfirpm(___)
```

### 说明

`b = cfirpm（n，f，fresp）` 返回长度为 +1 的 FIR 滤波器，其最佳近似值为 由函数返回的频率下所需的响应，该函数由其函数句柄*fresp*调用。

`B = cfirpm（n，f，fresp，w）` 使用指定的权重对每个频率的拟合进行加权

`b = cfirpm（n，f，a）` 指定中波段边缘的振幅。

`B = CFIRPM（N，F，A，W）` 应用一组可选的正权重，每个波段一个，用于优化。如果 您未指定 ，该函数将权重设置为 Unity。

`b = cfirpm(___,sym)`对设计的脉冲响应施加了对称约束。除了指定前面任何一个的输入组合语法。

`b = cfirpm(___,debug)`在滤波器设计过程中显示或隐藏中间结果。
`b = cfirpm(___,lgrid)`控制频率网格的密度。

`b = cfirpm(___,'skip_stage2')`禁用第二阶段优化算法，该算法仅在函数确定最佳解未被执行时执行通过标准错误交换达到。禁用此算法可以提高计算速度，但会降低准确性。默认情况下，启用第二阶段优化。

`[b,delta] = cfirpm(___)`返回最大纹波高度。

`[b,delta,opt] = cfirpm(___)`返回由函数计算的可选结果。

### 输入参数

n — 滤波顺序
实正标量

滤波器顺序，指定为实数正标量。

f— 元素在 [–1， 1] 范围内的归一化频率点
实值向量
归一化频率点，指定为实值向量，其中元素位于范围 [–1， 1]，其中 1 对应于归一化的奈奎斯特频率。频率必须按递增顺序排列，并且必须具有均匀的长度。这对于k奇数，频带范围为 （*k*+1）。k奇数的区间 （*k*+1） 到 （*k*+2） 是过渡带。

fresp— 频率响应
功能句柄
频率响应，指定为函数句柄。

a— 所需振幅
矢量
在 f 中指定的点处的所需振幅，指定为向量。对于k奇数，点对（k）和（k+1）之间的频率所需的振幅是直线 连接点的段 （（k），（k）） 和 （（k+1），（k+1））。

w— 权重
矢量
在 f 中指定的点处的所需振幅，指定为实值向量。的长度是 f 长度的一半，因此每个波段正好存在一个权重。如果未指定 ，则函数将权重设置为 unity。

sym— 对称约束
'none' （默认值）
对滤波器设计的脉冲响应施加的对称性约束，指定作为以下值之一：
'none'— 不施加对称约束。此选项是 如果将任何负频带频率传递给函数，或者 Fresp 未提供默认值，则为 default。
'even'— 施加真实甚至冲动的反应。这选项是 Highpass、Lowpass、AllPass、BandPass、Bandstop、 反向 Sinc 和多频段设计。
'odd'— 施加真实而奇怪的脉冲反应。此选项是 Hilbert 和微分器设计的默认值。
'real'— 对频率施加共轭对称性响应。
如果指定以外的值，则必须指定仅正频率上的频带边缘（填充负频率区域从对称性）。如果未指定 ，则函数将查询默认设置。任何用户提供的函数在作为筛选器顺序传递时都必须返回有效选项。

debug— 中间结果
显示“关闭”（默认）
在滤波器设计期间显示中间结果

lgrid— 频率网格
密度 25 （默认） |整数的 cell 数组
频率网格的密度，指定为整数的数组。频率网格大致具有频率点。

### 输出参量

b— 滤波器系数
行向量
滤波器系数，以长度为 n+1 的行向量返回。

delta— 最大纹波高度
标量
最大纹波高度，以标量形式返回。

opt— 可选的结果
结构
由函数计算的可选结果，以包含这些字段opt.fgrid/des/wt/H/error/iextr/fextr的结构。



## Python函数描述：cfirpm

函数来源：自定义

### cfirpm函数定义：

def cfirpm(numtaps, bands, ftype, fs=2):
    ...
    return taps

```python
import numpy as np
from scipy.signal import remez, freqz
import matplotlib.pyplot as plt

def cfirpm(numtaps, bands, ftype, fs=2):
    # 对 bands 进行正规化，使其在 [0, fs/2] 范围内
    bands = np.array(bands)
    bands = (bands + 1) / 2  # 将 [-1, 1] 映射到 [0, 1]

    # 根据给定的带设置 desired amplitude 和 weights
    # MATLAB 中给出了三个带（一个通带和两个阻带）
    # 我们需要在 Python 中对应设置它们
    # 但由于 Python 中的 remez 需要带的 '开始' 和 '结束'，我们有 4 个带边界
    # 而且 'desired' 数组需要对每个频带提供一个目标幅度值
    desired = [0, 1, 0]  # Passband, Stopband, Passband
    weight = [1, 1, 1]

    # 使用 remez 计算滤波器系数
    # numtaps+1 是因为 remez 设计长度为 numtaps+1 的滤波器
    taps = remez(numtaps+1, bands, desired, weight, Hz=fs)
    return taps

# 设计滤波器
b = cfirpm(30, [-1, -0.5, -0.4, 0.7, 0.8, 1], 'lowpass')

# 计算频率响应
w, h = freqz(b, worN=8192, whole=True)

# 绘制幅度响应
plt.subplot(2, 1, 1)
plt.plot(w / np.pi, 20 * np.log10(abs(h)), 'b')
plt.title('Lowpass Filter Frequency Response')
plt.ylabel('Amplitude [dB]')
plt.grid(True)

# 绘制相位响应
plt.subplot(2, 1, 2)
plt.plot(w / np.pi, np.unwrap(np.angle(h)) * 180 / np.pi, 'g')
plt.ylabel('Phase (degrees)')
plt.xlabel('Normalized Frequency (x π rad/sample)')
plt.grid(True)

plt.tight_layout()
plt.show()
```


这段代码定义了一个名为 `cfirpm` 的函数，其目的是使用 Parks-McClellan 算法（一种 Remez 交换算法的变体）设计一个具有特定频带特性的 FIR（有限冲击响应）滤波器。Parks-McClellan 算法是一种优化技术，它通过最小化一个频率函数上的最大误差来计算滤波器的系数。这种方法特别适合于需要精确控制频带特性（如通带和阻带的边界）的应用场景。通过这种方法，我们能够精确地定义每个频带内的期望幅度响应，以及相应的权重，从而设计出满足特定规格要求的滤波器。

### 参数
- **numtaps** (`int`): 滤波器的抽头数，决定了滤波器的阶数。
- **bands** (`list` of `float`): 设计滤波器时的频带边界，必须是一个从负到正的顺序列表。
- **ftype** (`str`): 滤波器类型，此处作为未使用的参数，保留以便将来可能的功能扩展。
- **fs** (`float`, 可选): 采样频率，默认值为2，用于指定滤波器设计中的正规化频率。

### 返回值
- **taps** (`numpy.ndarray`): 计算得到的滤波器抽头系数数组。

### 注意事项
- 确保 `bands` 数组中的元素严格按从小到大的顺序排列，且界限值在 [-1, 1] 内。

### 函数工作原理
1. 该函数使用 `remez` 函数实现 FIR 滤波器的设计。`bands` 参数需正规化到 [0, fs/2] 的范围内，`desired` 和 `weight` 数组用于指定带的幅度响应和权重。

### 使用场景
适用于需要定制不同阻带和通带频率特性的场合，如音频处理、信号去噪等。

### 改进建议
- 当前实现假设 `bands` 数组长度适用于三个带区域，未来可以扩展支持更多或更少的带设置。
- 未来版本中可以添加更复杂的权重和幅度设计功能，以支持更复杂的滤波器需求。



## Prompt 1 ： 生成 Python cfirpm 函数

参考下面MATLAB示例：设计一个 30 阶线性相位低通滤波器。显示其幅度和相位响应。

以及MATLAB代码的cfirpm函数：

```
b = cfirpm(30,[-1 -0.5 -0.4 0.7 0.8 1],@lowpass);
freqz(b,1,[],"whole")
```

请完成Python语言cfirpm函数的编写，便于类似应用调用。

### LLM 生成 ： Python cfirpm 函数

这是cfirpm函数定义
def cfirpm(numtaps, bands, ftype, fs=2):
    ...
    return taps

    import numpy as np
    from scipy.signal import remez, freqz
    import matplotlib.pyplot as plt
    
    def cfirpm(numtaps, bands, ftype, fs=2):
        # 对 bands 进行正规化，使其在 [0, fs/2] 范围内
        bands = np.array(bands)
        bands = (bands + 1) / 2  # 将 [-1, 1] 映射到 [0, 1]
    
        # 根据给定的带设置 desired amplitude 和 weights
        # MATLAB 中给出了三个带（一个通带和两个阻带）
        # 我们需要在 Python 中对应设置它们
        # 但由于 Python 中的 remez 需要带的 '开始' 和 '结束'，我们有 4 个带边界
        # 而且 'desired' 数组需要对每个频带提供一个目标幅度值
        desired = [0, 1, 0]  # Passband, Stopband, Passband
        weight = [1, 1, 1]
    
        # 使用 remez 计算滤波器系数
        # numtaps+1 是因为 remez 设计长度为 numtaps+1 的滤波器
        taps = remez(numtaps+1, bands, desired, weight, Hz=fs)
        return taps
    
    # 设计滤波器
    b = cfirpm(30, [-1, -0.5, -0.4, 0.7, 0.8, 1], 'lowpass')
    
    # 计算频率响应
    w, h = freqz(b, worN=8192, whole=True)
    
    # 绘制幅度响应
    plt.subplot(2, 1, 1)
    plt.plot(w / np.pi, 20 * np.log10(abs(h)), 'b')
    plt.title('Lowpass Filter Frequency Response')
    plt.ylabel('Amplitude [dB]')
    plt.grid(True)
    
    # 绘制相位响应
    plt.subplot(2, 1, 2)
    plt.plot(w / np.pi, np.unwrap(np.angle(h)) * 180 / np.pi, 'g')
    plt.ylabel('Phase (degrees)')
    plt.xlabel('Normalized Frequency (x π rad/sample)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

## Prompt 2 ： 应用生成的 Python cfirpm.py 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：

注意：
(1)线条采用不同颜色方便区分。

**Matlab代码**
```
b = cfirpm(30,[-1 -0.5 -0.4 0.7 0.8 1],@lowpass);
freqz(b,1,[],"whole")
```

这是Python envelope.py
```
import numpy as np
from scipy.signal import remez, freqz
import matplotlib.pyplot as plt

def cfirpm(numtaps, bands, ftype, fs=2):
    # 对 bands 进行正规化，使其在 [0, fs/2] 范围内
    bands = np.array(bands)
    bands = (bands + 1) / 2  # 将 [-1, 1] 映射到 [0, 1]

    # 根据给定的带设置 desired amplitude 和 weights
    # MATLAB 中给出了三个带（一个通带和两个阻带）
    # 我们需要在 Python 中对应设置它们
    # 但由于 Python 中的 remez 需要带的 '开始' 和 '结束'，我们有 4 个带边界
    # 而且 'desired' 数组需要对每个频带提供一个目标幅度值
    desired = [0, 1, 0]  # Passband, Stopband, Passband
    weight = [1, 1, 1]

    # 使用 remez 计算滤波器系数
    # numtaps+1 是因为 remez 设计长度为 numtaps+1 的滤波器
    taps = remez(numtaps+1, bands, desired, weight, Hz=fs)
    return taps

# 设计滤波器
b = cfirpm(30, [-1, -0.5, -0.4, 0.7, 0.8, 1], 'lowpass')

# 计算频率响应
w, h = freqz(b, worN=8192, whole=True)

# 绘制幅度响应
plt.subplot(2, 1, 1)
plt.plot(w / np.pi, 20 * np.log10(abs(h)), 'b')
plt.title('Lowpass Filter Frequency Response')
plt.ylabel('Amplitude [dB]')
plt.grid(True)

# 绘制相位响应
plt.subplot(2, 1, 2)
plt.plot(w / np.pi, np.unwrap(np.angle(h)) * 180 / np.pi, 'g')
plt.ylabel('Phase (degrees)')
plt.xlabel('Normalized Frequency (x π rad/sample)')
plt.grid(True)

plt.tight_layout()
plt.show()
```


### LLM 生成 ： 

下面这是调用程序
```
import matplotlib.pyplot as plt
from scipy.signal import freqz
from cfirpm import cfirpm  # 确保 cfirpm.py 在 Python 的模块搜索路径上
import numpy as np
# 使用 cfirpm 函数设计滤波器
b = cfirpm(30, [-1, -0.5, -0.4, 0.7, 0.8, 1], 'lowpass')

# 使用 freqz 获取滤波器的频率响应
w, h = freqz(b, worN=8192, whole=True)

# 绘制滤波器的幅度响应
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w / np.pi, 20 * np.log10(abs(h)), 'b')
plt.title('Lowpass Filter Frequency Response')
plt.ylabel('Amplitude [dB]')
plt.grid()

# 绘制滤波器的相位响应
plt.subplot(2, 1, 2)
angles = np.unwrap(np.angle(h))
plt.plot(w / np.pi, angles, 'g')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Angle (radians)')
plt.grid()
plt.show()
```




