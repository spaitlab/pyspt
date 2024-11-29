# 信号处理仿真与应用 - 信号生成和预处理 - 波形生成

## MATLAB函数描述：gmonopuls

函数来源：[MATLAB gmonopuls](https://ww2.mathworks.cn/help/signal/ref/gmonopuls.html)

### 语法

y = gmonopuls(t,fc)
tc = gmonopuls('cutoff',fc)

### 说明

y = gmonopuls(t,fc) 返回中心频率为fc的单位振幅高斯单脉冲在阵列t所示时间的采样。
tc = gmonopuls('cutoff',fc) 返回脉冲的最大和最小振幅之间的持续时间。

### 输入参数

t — 时间值
向量
计算单位振幅高斯单脉冲的时间值，以向量表示。

fc — 中心频率
1000(默认)|实正标量
中心频率，用实正标量表示，以赫兹表示。默认情况下，fc = 1000hz。

### 输出参量

y - 单脉冲
向量
单位振幅的单脉冲，以矢量形式返回。

tc — 持续时间
标量
脉冲最大和最小振幅之间的持续时间，以标量形式返回。



## Python函数描述：gmonopuls

函数来源：自定义

### 高斯单脉冲函数定义：

def gmonopuls(t, fc):
    tau = 1 / (2 * np.pi * fc)
    a = 1 / (2 * tau**2)
    y = t * np.exp(-a * t**2)
    peak_value = np.max(y)
    y = y / (peak_value+1e-15)
    return y


这段代码定义了一个名为 `gmonopuls` 的函数，其目的是生成高斯单极性脉冲。高斯单极性脉冲是一种特定形式的信号，其在零时刻以前为零，并且其形状由高斯函数调制。这种脉冲类型常用于雷达系统和其他通信系统中，因为它的频谱特性和时间局限性很适合脉冲压缩技术。

### 参数
- `t`: 时间数组，代表信号的时间基线，通常是等间隔的。
- `fc`: 载波频率，定义脉冲中心频率的参数，通常以赫兹(Hz)为单位。

### 返回值
- `y`: 归一化的高斯单极性脉冲信号。该信号在零时刻之前为零，随后快速上升并达到峰值，然后迅速衰减。

### 注意事项
- 高斯单极性脉冲主要适用于需要脉冲压缩和较宽带宽的应用，如雷达和通信。
- 归一化保证了在不同的系统或应用中信号的幅度是可比的，但在实际应用中可能需要根据具体情况调整归一化处理。

### 函数工作原理
1. 计算时间常数 `tau`，它与载波频率 `fc` 成反比，公式为 \( \tau = \frac{1}{2 \pi f_c} \)。
2. 使用 `tau` 计算控制脉冲宽度和形状的系数 `a`，公式为 \( a = \frac{1}{2 \tau^2} \)。
3. 生成高斯单极性脉冲，公式为 \( y = t \cdot e^{-a t^2} \)，其中仅当 \( t \geq 0 \) 时脉冲不为零。
4. 找到脉冲的峰值，并通过该峰值对脉冲进行归一化，确保峰值为1，以便于进行幅度比较和处理。

### 使用场景
高斯单极性脉冲由于其频谱特性和时间局限性，非常适合于雷达和通信系统中的脉冲压缩技术，可以有效地提高系统的分辨率和检测性能。此外，它也常用于科学研究和工程应用中，用于生成具有特定频率和时间特性的测试信号。

### 改进建议
- 根据实际应用需求，可以考虑加入更多的参数调节功能，如调整脉冲的时间宽度和频率特性，以便更好地适配特定的系统或环境。
- 考虑实现更多的脉冲类型，如双极性高斯脉冲或其他形状的脉冲，以扩展函数的应用范围。
- 对函数的数值稳定性和计算效率进行优化，确保在处理大规模数据或高速处理需求时，性能和准确性能得到保证。


## Prompt 1 ： 生成 Python gmonopuls 函数

参考下面MATLAB代码的gmonopuls函数
```
fc = 2e9;
fs = 100e9;

tc = gmonopuls('cutoff',fc);
t  = -2*tc:1/fs:2*tc;

y = gmonopuls(t,fc);
```

请完成Python语言gmonopuls函数的编写，便于类似应用调用。

### LLM 生成 ： Python gmonopuls 函数

这是高斯单脉冲函数定义
def gmonopuls(t, fc):
    tau = 1 / (2 * np.pi * fc)
    a = 1 / (2 * tau**2)
    y = t * np.exp(-a * t**2)
    peak_value = np.max(y)
    y = y / (peak_value+1e-15)
    return y



## Prompt 2 ： 应用生成的 Python gmonopuls 函数做MATLAB示例对应

这是matlab代码
```
fc = 2e9;
fs = 100e9;

tc = gmonopuls('cutoff',fc);
t  = -2*tc:1/fs:2*tc;

y = gmonopuls(t,fc);

sg = 1/(2*pi*fc);

ys = exp(1/2)*t/sg.*exp(-(t/sg).^2/2);

plot(t,y,t,ys,'.')
legend('gmonopuls','Definition')
```

这是高斯单脉冲函数定义
def gmonopuls(t, fc):
    tau = 1 / (2 * np.pi * fc)
    a = 1 / (2 * tau**2)
    y = t * np.exp(-a * t**2)
    peak_value = np.max(y)
    y = y / (peak_value+1e-15)
    return y


请给出应用高斯单脉冲函数gmonopuls给出第一段代码的Python程序。

### LLM 生成 ： 
```
import numpy as np
import matplotlib.pyplot as plt

# 定义参数
fc = 2e9  # 载波频率
fs = 100e9  # 采样频率

# 计算截止时间，对于高斯脉冲，我们可以用类似的形式来估算
tc = np.sqrt(np.log(2)) / (np.pi * fc)

# 生成时间数组
t = np.arange(-2*tc, 2*tc, 1/fs)

# 手动创建高斯单极性脉冲
def gmonopuls(t, fc):
    tau = 1 / (2 * np.pi * fc)
    a = 1 / (2 * tau**2)
    y = t * np.exp(-a * t**2)
    peak_value = np.max(y)
    y = y / (peak_value+1e-15)
    return y

y = gmonopuls(t, fc)

sg = 1 / (2 * np.pi * fc)  # 高斯标准差

# 使用给出的定义创建另一个高斯脉冲
ys = (np.exp(1/2) * t / sg) * np.exp(-(t / sg) ** 2 / 2)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(t, y, label='gmonopuls')
plt.plot(t, ys, '.', label='Definition')
plt.title('Comparison of Gaussian Pulses')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
```

## Prompt 3 ： 应用生成的 Python gmonopuls 函数做MATLAB示例对应

这是matlab代码
```
fc = 2e9;
fs = 100e9;

tc = gmonopuls('cutoff',fc);
D = ((0:2)*7.5+2.5)*1e-9;

t  = 0:1/fs:150*tc;
yp = pulstran(t,D,'gmonopuls',fc);

plot(t,yp)
```

这是高斯单脉冲函数定义
def gmonopuls(t, fc):
    tau = 1 / (2 * np.pi * fc)
    a = 1 / (2 * tau**2)
    y = t * np.exp(-a * t**2)
    peak_value = np.max(y)
    y = y / (peak_value+1e-15)
    return y


请给出应用高斯单脉冲函数gmonopuls给出第一段代码的Python程序。

### LLM 生成 ： 
```
import numpy as np

fc = 2e9 # 载波频率
fs = 100e9  # 采样频率


# 计算截止时间，使用之前讨论的方法来估算
tc = np.sqrt(np.log(2)) / (np.pi * fc)

# 创建时间数组，从 2.5ns 开始，每次增加 7.5ns，直到包括 17.5ns
D = (np.arange(3) * 7.5 + 2.5) * 1e-9

import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.interpolate import interp1d

# 生成时间向量
t = np.arange(0, 150 * tc, 1/fs)

# 使用插值法生成脉冲
def generate_pulses(t, D, pulse_func, fc):
    result = np.zeros_like(t)
    for d in D:
        shifted_t = t - d
        pulse = pulse_func(shifted_t, fc)
        result += pulse
    return result

# 生成脉冲列
yp = generate_pulses(t, D, gmonopuls, fc)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(t, yp)
plt.title('Pulse Train Generated with gmonopuls')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```



