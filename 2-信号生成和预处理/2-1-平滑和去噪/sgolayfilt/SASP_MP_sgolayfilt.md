# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：sgolayfilt

函数来源：[MATLAB sgolayfilt](https://ww2.mathworks.cn/help/signal/ref/sgolayfilt.html)

### 语法

y = sgolayfilt(x,order,framelen)
y = sgolayfilt(x,order,framelen,weights)
y = sgolayfilt(x,order,framelen,weights,dim)

### 说明

y = sgolayfilt(x,order,framelen) 对向量 x 中的数据应用多项式阶数为 order、帧长度为 framelen 的萨维茨基-戈雷有限冲激响应 (FIR) 平滑滤波器。如果 x 是矩阵，则 sgolayfilt 对每列进行运算。
y = sgolayfilt(x,order,framelen,weights) 指定在最小二乘最小化过程中要使用的加权向量。
y = sgolayfilt(x,order,framelen,weights,dim) 指定滤波器沿其运算的维度。

### 输入参数

x — 输入信号
向量 | 矩阵
输入信号，指定为向量或矩阵。
数据类型: single | double

order — 多项式阶数
正整数
多项式阶数，指定为正整数。order 必须小于 framelen。如果 order = framelen - 1，则滤波器不会产生平滑效果。
数据类型: single | double

framelen — 帧长度
正奇数
帧长度，指定为正奇数。
数据类型: single | double

weights — 加权数组
正实数向量 | 正实数矩阵
加权数组，指定为长度为 framelen 的正实数向量或矩阵。
数据类型: single | double

dim — 要沿其滤波的维度
正整数标量
要沿其滤波的维度，指定为正整数标量。默认情况下，sgolayfilt 沿 x 的大小大于 1 的第一个维度进行运算。
数据类型: single | double

### 输出参量

y — 滤波后的信号
向量 | 矩阵
滤波后的信号，以向量或矩阵形式返回。



## Python函数描述：

函数来源：[scipy.signal.savgol_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)

### 萨维茨基-戈雷滤波函数定义：

def sgolayfilt(x,order,framelen) :
    """
    Apply a Savitzky-Golay filter to an array.

    Parameters:
    x (array_like): Input signal. Vector or matrix.
    order (int): Polynomial order. Positive integer. 'order' must be less than 'framelen'. If 'order' = 'framelen' - 1, the filter does not produce any smoothing effect.
    framelen (int): Frame length. Positive odd integer.

    Returns:
    y (array_like): Filtered signal. Vector or matrix.

    Note: Details on the mode options:
    ‘mirror’:Repeats the values at the edges in reverse order. The value closest to the edge is not included.
    ‘nearest’:The extension contains the nearest input value.
    ‘constant’:The extension contains the value given by the cval argument.
    ‘wrap’:The extension contains the values from the other end of the array.




这段代码定义了一个名为 sgolayfilt 的函数，用于应用萨维茨基-戈雷有限冲激响应 (Savitzky-Golay FIR) 平滑滤波器。函数接受输入信号 x、多项式阶数 order、帧长度 framelen。函数返回滤波后的信号 y，函数的参数说明包括输入参数和输出参数。

### 参数
- `x`: 输入信号，可以是向量或矩阵。
- `order`: 多项式阶数，正整数。'order' 必须小于 'framelen'。如果 'order' = 'framelen' - 1，则滤波器不会产生平滑效果。
- `framelen`: 帧长度，正奇数。

### 返回值
- `y`: 滤波后的信号，可以是向量或矩阵。

### 注意事项
- 关于模式选项的详细信息：
    ‘mirror’：以相反顺序重复边缘的值。最靠近边缘的值不包括在内。
    ‘nearest’：扩展包含最近的输入值。
    ‘constant’：扩展包含由cval参数给出的值。
    ‘wrap’：扩展包含数组的另一端的值。

### 函数工作原理
1. 根据输入的信号 x、多项式阶数 order 和窗口长度 framelen，应用Savitzky-Golay滤波器。
2. 根据给定的模式选项对信号进行边界扩展。
3. 使用滤波器的系数对信号进行卷积操作，得到滤波后的信号 y。


### 使用场景
Savitzky-Golay滤波器适用于多种领域，包括生物信号处理（如心电图和脑电图信号的去噪）、传感器数据处理（如加速度计和温度传感器数据的平滑）、光谱分析（如红外光谱和质谱分析中的光谱曲线平滑）、图像处理（如图像噪声去除）、信号分析（如频谱特性分析和特征提取）。

### 改进建议
- 增加输入参数检查： 在函数中添加输入参数的类型和范围检查，以确保输入参数的有效性和一致性，提高函数的鲁棒性。
- 优化滤波器设计： 考虑实现更高级的Savitzky-Golay滤波器设计算法，如改进的最小二乘法拟合或基于奇异值分解的方法，以提高滤波器的性能和效率。
- 支持更多的模式选项： 考虑添加更多的模式选项，以满足不同情况下的边界扩展需求，例如边界值填充、周期性扩展等。



## Prompt  ： scipy.signal.savgol_filter做MATLAB示例对应


```
% 生成一个随机信号并使用 sgolayfilt 对其进行平滑处理。指定多项式阶数为 3，帧长度为 11。绘制原始信号和经过平滑处理的信号。
order = 3;
framelen = 11;

lx = 34;
x = randn(lx,1);

sgf = sgolayfilt(x,order,framelen);

plot(x,':')
hold on
plot(sgf,'.-')
legend('signal','sgolay')
```

采用Python语言实现的绘制原始信号和经过平滑处理的信号，
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, convolve

# 参数定义
order = 3
framelen = 11
lx = 34

# 生成随机信号
x = np.random.randn(lx)

# 使用 sgolayfilt 函数进行滤波处理
sgf = savgol_filter(x, framelen, order)

# 绘制原始信号和经过滤波处理后的信号
plt.plot(x, ':', label='signal')
plt.plot(sgf, '.-', label='sgolay')
plt.legend()
plt.show()

```




