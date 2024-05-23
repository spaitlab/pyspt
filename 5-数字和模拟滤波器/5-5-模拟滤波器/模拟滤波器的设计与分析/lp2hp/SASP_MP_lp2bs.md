# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：lp2hp 

函数来源：[MATLAB lp2hp](https://ww2.mathworks.cn/help/signal/ref/lp2hp.html?s_tid=doc_ta)

### 语法
[bt,at] = lp2hp(b,a,Wo)

[At,Bt,Ct,Dt] = lp2hp(A,B,C,D,Wo)
### 说明

[bt,at] = lp2hp(b,a,Wo) 将由多项式系数（由行向量 b 和 a 指定）表示的模拟低通滤波器原型转换为具有截止角频率 Wo 的高通模拟滤波器。输入系统必须是模拟滤波器原型。

[At,Bt,Ct,Dt] = lp2hp(A,B,C,D,Wo) 将由矩阵 A, B, C, 和 D 指定的连续时间状态空间低通滤波器原型转换为具有截止角频率 Wo 的高通模拟滤波器。输入系统必须是模拟滤波器原型。
### 输入参数
b, a — 原型分子和分母系数
行向量
原型分子和分母系数，以行向量的形式指定。
数据类型：single | double

A, B, C, D — 原型状态空间表示
矩阵
原型状态空间表示，以矩阵形式指定。
数据类型：single | double

Wo — 截止角频率
标量
截止角频率，以标量形式指定。将截止角频率以 rad/s 为单位表示。
数据类型：single | double
### 输出参量
bt, at — 转换后的分子和分母系数
行向量
转换后的分子和分母系数，作为行向量返回。

At, Bt, Ct, Dt — 转换后的状态空间表示
矩阵
转换后的状态空间表示，作为矩阵返回。
## Python函数描述：lp2hp

函数来源：scipy.signal模块

### 将低通模拟滤波器转换为高通滤波器函数定义：
def lp2hp(b, a, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a highpass filter.

    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency, in
    transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed high-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed high-pass filter.


这段代码定义了一个名为 `lp2hp` 的函数，将低通模拟滤波器转换为高通滤波器。

### 参数

- `b` : 数组类型
分子多项式系数。

- `a` : 数组类型
分母多项式系数。

- `wo` : 浮点数
所需的截止频率，以角频率为单位（例如，rad/s）。
默认情况下不做改变。
默认值为 1。
### 返回值
- `b` : 数组类型
转换后的高通滤波器的分子多项式系数。

- `a` : 数组类型
转换后的高通滤波器的分母多项式系数。
### 注意事项
- lp2hp 函数基于模拟滤波器设计理论，主要适用于模拟域内的滤波器转换。对于数字信号处理应用，需要先设计模拟滤波器，然后通过适当的转换方法（如脉冲响应不变法或双线性变换）转换为数字滤波器。
- 该函数使用模拟滤波器的分子和分母系数进行高通滤波器的设计，不涉及数字滤波器设计参数，如采样频率。
### 函数工作原理
1. 接受模拟低通滤波器原型的分子 b 和分母 a 系数作为输入。
2. 根据提供的截止频率 wo，通过 s-plane 替换方法将低通滤波器的极点和零点映射到相应的高通滤波器位置。
3. 返回新的分子 bh 和分母 ah 系数，代表转换后的高通滤波器。
### 使用场景
lp2hp 函数适用于需要创建高通滤波器的场景，如在音频处理、生物医学信号处理和图像处理中去除低频噪声或基线漂移。

在模拟信号预处理阶段，该函数可以帮助设计满足特定截止频率要求的高通滤波器。

### 改进建议
- 可以考虑增加对数字滤波器设计的直接支持，提供从模拟滤波器到数字滤波器的无缝转换。
- 可以提供额外的参数，允许用户选择不同的滤波器转换方法，以改善在特定应用中的性能。


## Prompt 1 ： 应用 Python 的 lp2hp 函数做 MATLAB 示例对应

参考下面MATLAB代码
```
%设计截止频率为100 Hz、通带纹波为3 dB、阻带衰减为30 dB的五阶高通椭圆滤波器

%设计原型。将零极点增益输出转换为传递函数。

f = 100;

[ze,pe,ke] = ellipap(5,3,30);
[be,ae] = zp2tf(ze,pe,ke);
%将原型转换为高通滤波器。以 rad/s 为单位指定截止频率。

[bh,ah] = lp2hp(be,ae,2*pi*f);
%计算并绘制滤波器的频率响应。将归一化频率除以2π所以图的 x 轴是 Hz。

[hh,wh] = freqs(bh,ah,4096);

semilogx(wh/2/pi,mag2db(abs(hh)))
axis([10 400 -40 5])
grid
```

我们采用Python语言实现将低通模拟滤波器转换为高通滤波器
```
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

f = 100

# 设计原型滤波器
ze, pe, ke = signal.ellipap(5, 3, 30)
be, ae = signal.zpk2tf(ze, pe, ke)

# 将原型滤波器转换为高通滤波器
bh, ah = signal.lp2hp(be, ae, 2 * np.pi * f)

# 计算并绘制滤波器的频率响应
w, h = signal.freqs(bh, ah, worN=4096)

```

### LLM 生成 ： 调用程序实现将低通模拟滤波器转换为高通滤波器并展示出

```
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

f = 100

# 设计原型滤波器
ze, pe, ke = signal.ellipap(5, 3, 30)
be, ae = signal.zpk2tf(ze, pe, ke)

# 将原型滤波器转换为高通滤波器
bh, ah = signal.lp2hp(be, ae, 2 * np.pi * f)

# 计算并绘制滤波器的频率响应
w, h = signal.freqs(bh, ah, worN=4096)

# 绘制频率响应图
plt.semilogx(w / (2 * np.pi), 20 * np.log10(np.abs(h)))
plt.axis([10, 400, -40, 5])
plt.title('Highpass Elliptic Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.show()
```

