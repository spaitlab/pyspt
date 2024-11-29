# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：lp2bp 

函数来源：[MATLAB lp2bp](https://ww2.mathworks.cn/help/signal/ref/lp2bp.html?s_tid=doc_ta)

### 语法
[bt,at] = lp2bp(b,a,Wo,Bw)

[At,Bt,Ct,Dt] = lp2bp(A,B,C,D,Wo,Bw)

### 说明

[bt,at] = lp2bp(b,a,Wo,Bw) 将由多项式系数（由行向量 b 和 a 指定）表示的模拟低通滤波器原型转换为具有中心频率 Wo 和带宽 Bw 的带通滤波器。输入系统必须是模拟滤波器原型。

[At,Bt,Ct,Dt] = lp2bp(A,B,C,D,Wo,Bw) 将连续时间状态空间低通滤波器原型（由矩阵 A, B, C, 和 D 指定）转换为具有中心频率 Wo 和带宽 Bw 的带通滤波器。输入系统必须是模拟滤波器原型。

### 输入参数

b, a — 原型分子和分母系数
行向量
数据类型：single | double

A, B, C, D — 原型状态空间表示
矩阵
数据类型：single | double

Wo — 中心频率
标量
中心频率，以标量形式指定。对于具有下带边缘 w1 和上带边缘 w2 的滤波器，使用 Wo = sqrt(w1*w2)。Wo 的单位为 rad/s。
数据类型：single | double

Bw — 带宽
标量
带宽，以标量形式指定。对于具有下带边缘 w1 和上带边缘 w2 的滤波器，使用 Bw = w2–w1。Bw 的单位为 rad/s。
数据类型：single | double

### 输出参量

bt, at — 转换后的分子和分母系数
行向量
作为行向量返回的转换后的分子和分母系数。

At, Bt, Ct, Dt — 转换后的状态空间表示
矩阵
作为矩阵返回的转换后的状态空间表示。


## Python函数描述：lp2bp

函数来源：SciPy 库中的 scipy.signal 模块

### 低通模拟滤波器转换为带通的函数定义：

def lp2bp(b, a, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, in transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed band-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed band-pass filter.


这段代码定义了一个名为 `lp2bp` 的函数，用于将模拟低通滤波器原型转换为带通滤波器。

### 参数

- `b` : 数组类型
分子多项式系数。
- `a` : 数组类型
分母多项式系数。
- `wo` : 浮点数
所需的通带中心频率，以角频率为单位（例如，rad/s）。
默认情况下不做改变。
- `bw` : 浮点数
所需的通带宽度，以角频率为单位（例如，rad/s）。
默认值为 1。
### 返回值
- `b` : 数组类型
转换后的带通滤波器的分子多项式系数。
- `a` : 数组类型
转换后的带通滤波器的分母多项式系数。
### 注意事项
- lp2bp 函数基于模拟滤波器设计理论，主要适用于模拟域内的滤波器转换。对于数字信号处理应用，需要先设计模拟滤波器，然后通过适当的转换方法（如双线性变换或脉冲响应不变法）转换为数字滤波器。
- 该函数使用 Tustin 方法（s-domain 的一种映射）进行滤波器转换，这可能导致与理想带通滤波器特性存在偏差，特别是在处理具有严格带宽要求的高频信号时。
### 函数工作原理
1. 接受模拟低通滤波器的分子 b 和分母 a 系数作为输入。
2. 利用提供的中心频率 wo 和带宽 bw，通过 s-plane 替换方法将低通滤波器的极点和零点映射到相应的带通滤波器位置。
3. 返回新的分子 bt 和分母 at 系数，代表转换后的带通滤波器。

### 使用场景
lp2bp 函数适用于需要将已有的模拟低通滤波器原型转换为带通滤波器的场景，如在通信系统、信号处理和音频处理中设计带通滤波器。

在模拟信号预处理阶段，该函数可以帮助设计满足特定中心频率和带宽要求的带通滤波器。
### 改进建议
- 如果用户需要在数字信号处理环境中使用带通滤波器，可以提供额外的指导或函数来展示如何从 lp2bp 的输出转换为数字滤波器。
- 可以考虑集成预畸变处理，以更准确地映射模拟滤波器的频率响应到数字域。


## Prompt 1 ： 应用 Python 的 lp2bp 函数做 MATLAB 示例对应

参考下面MATLAB代码
```
%设计一个 14 阶低通巴特沃斯模拟滤波器原型。

n = 14;
[z,p,k] = buttap(n);
%将原型转换为传递函数形式。显示其幅度和频率响应。
[b,a] = zp2tf(z,p,k);
freqs(b,a)
%将原型转换为通带从30 Hz到100 Hz的带通滤波器。 指定中心频率和带宽（以rad/s为单位）。
fl = 30;
fh = 100;

Wo = 2*pi*sqrt(fl*fh); % center frequency
Bw = 2*pi*(fh-fl); % bandwidth

[bt,at] = lp2bp(b,a,Wo,Bw);
%显示变换后的滤波器的幅度和频率响应。
freqs(bt,at)
```

我们采用Python语言实现低通滤波器变换带通滤波器
```
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

n = 14
z, p, k = signal.buttap(n)

b, a = signal.zpk2tf(z, p, k)
w, h = signal.freqs(b, a)

fl = 30
fh = 100

Wo = 2 * np.pi * np.sqrt(fl * fh)
Bw = 2 * np.pi * (fh - fl)

bt, at = signal.lp2bp(b, a, Wo, Bw)
w, h = signal.freqs(bt, at)

```


### LLM 生成 ： 调用程序实现低通滤波器变换带通滤波器


```
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

n = 14
z, p, k = signal.buttap(n)

b, a = signal.zpk2tf(z, p, k)
w, h = signal.freqs(b, a)

plt.figure()
plt.semilogx(w, 20 * np.log10(np.abs(h)))
plt.title('Analog Butterworth Filter Frequency Response')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()

plt.figure()
plt.semilogx(w, np.angle(h, deg=False))
plt.title('Analog Butterworth Filter Phase Response')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.show()

fl = 30
fh = 100

Wo = 2 * np.pi * np.sqrt(fl * fh)
Bw = 2 * np.pi * (fh - fl)

bt, at = signal.lp2bp(b, a, Wo, Bw)
w, h = signal.freqs(bt, at)

plt.figure()
plt.semilogx(w, 20 * np.log10(np.abs(h)))
plt.title('Bandpass Filter Frequency Response')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()

plt.figure()
plt.semilogx(w, np.angle(h, deg=False))
plt.title('Bandpass Filter Phase Response')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.show()
```

