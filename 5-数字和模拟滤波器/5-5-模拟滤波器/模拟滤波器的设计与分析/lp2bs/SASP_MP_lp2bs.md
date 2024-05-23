# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：lp2bs 

函数来源：[MATLAB lp2bs](https://ww2.mathworks.cn/help/signal/ref/lp2bs.html?s_tid=doc_ta)

### 语法
[bt,at] = lp2bs(b,a,Wo,Bw)

[At,Bt,Ct,Dt] = lp2bs(A,B,C,D,Wo,Bw)
### 说明

[bt,at] = lp2bs(b,a,Wo,Bw) 将由多项式系数（由行向量 b 和 a 指定）表示的模拟低通滤波器原型转换为具有中心频率 Wo 和带宽 Bw 的带阻滤波器。输入系统必须是模拟滤波器原型。

[At,Bt,Ct,Dt] = lp2bs(A,B,C,D,Wo,Bw) 将连续时间状态空间低通滤波器原型（由矩阵 A, B, C, 和 D 指定）转换为具有中心频率 Wo 和带宽 Bw 的带阻滤波器。输入系统必须是模拟滤波器原型。

### 输入参数

b, a — 原型分子和分母系数
行向量
原型分子和分母系数，指定为行向量。b 和 a 指定了原型的分子和分母系数，按照 s 的降幂排列：
数据类型：single | double

A, B, C, D — 原型状态空间表示
矩阵
原型状态空间表示，指定为矩阵。状态空间矩阵通过以下方式关联状态向量 x、输入 u 和输出 y：
数据类型：single | double

Wo — 中心频率
标量
中心频率，指定为标量。对于具有下带边缘 w1 和上带边缘 w2 的滤波器，使用 Wo = sqrt(w1*w2)。Wo 的单位为 rad/s。
数据类型：single | double

Bw — 带宽
标量
带宽，指定为标量。对于具有下带边缘 w1 和上带边缘 w2 的滤波器，使用 Bw = w2–w1。带宽的单位为 rad/s。
数据类型：single | double

### 输出参量
bt, at — 转换后的分子和分母系数
行向量
转换后的分子和分母系数，作为行向量返回。

At, Bt, Ct, Dt — 转换后的状态空间表示
矩阵
转换后的状态空间表示，作为矩阵返回。
## Python函数描述：lp2bs

函数来源：scipy.signal模块

### 低通模拟滤波器转换为带阻函数定义：
def lp2bs(b, a, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, in transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed band-stop filter.
    a : array_like
        Denominator polynomial coefficients of the transformed band-stop filter.

    See Also
    --------
    lp2lp, lp2hp, lp2bp, bilinear
    lp2bs_zpk

这段代码定义了一个名为 `lp2bs` 的函数，用于将低通滤波器转换为带阻滤波器.

### 参数
- `b` : 数组类型
分子多项式系数。

- `a` : 数组类型
分母多项式系数。

- `wo` : 浮点数
所需的阻带中心频率，以角频率为单位（例如，rad/s）。
默认情况下不做改变。

- `bw` : 浮点数
所需的阻带宽度，以角频率为单位（例如，rad/s）。
默认值为 1。
### 返回值
- `b` : 数组类型
转换后的带阻滤波器的分子多项式系数。

- `a` : 数组类型
转换后的带阻滤波器的分母多项式系数
### 注意事项
- lp2bs 函数基于模拟滤波器设计理论，主要适用于模拟域内的滤波器转换。对于数字信号处理应用，需要先设计模拟滤波器，然后通过适当的转换方法转换为数字滤波器。
- 该函数使用模拟滤波器的分子和分母系数进行带阻滤波器的设计，不涉及数字滤波器设计参数，如采样频率。
### 函数工作原理
1. 接受模拟低通滤波器原型的分子 b 和分母 a 系数作为输入。
2. 根据提供的阻带中心频率 wo 和阻带宽度 bw，通过数学变换将低通滤波器的极点和零点映射到相应的带阻滤波器位置。
3. 返回新的分子 bt 和分母 at 系数，代表转换后的带阻滤波器。
### 使用场景
lp2bs 函数适用于需要创建带阻滤波器的场景，如在通信系统、信号处理和音频处理中去除特定频段的干扰。

在模拟信号预处理阶段，该函数可以帮助设计满足特定阻带中心频率和宽度要求的带阻滤波器。
### 改进建议
- 可以考虑集成预畸变处理，以更准确地映射模拟滤波器的频率响应到数字域。
- 提供对输入参数的验证和错误提示，确保 b 和 a 系数以及 wo 和 bw 的值适合进行带阻滤波器转换。


## Prompt 1 ： 应用 Python 的 lp2bs 函数做 MATLAB 示例对应

参考下面MATLAB代码
```
% 设计一个 10 阶低通巴特沃斯模拟滤波器原型。
n = 10;
[z, p, k] = buttap(n);

% 将原型转换为传递函数形式，并显示其幅度和频率响应。
[b, a] = zp2tf(z, p, k);
figure; % 创建一个新的图形窗口
freqs(b, a); % 显示低通滤波器的幅度和频率响应

% 将原型转换为阻带从 20 Hz 到 60 Hz 的带阻滤波器。指定中心频率和带宽（以 rad/s 为单位）。
fl = 20;
fh = 60;
Wo = 2*pi*sqrt(fl*fh); % 中心频率
Bw = 2*pi*(fh-fl); % 带宽

[bt, at] = lp2bs(b, a, Wo, Bw);

% 显示变换后的滤波器的幅度和频率响应。
figure; % 创建另一个新的图形窗口
freqs(bt, at); % 显示带阻滤波器的幅度和频率响应
```

我们采用Python语言实现将10 阶低通巴特沃斯模拟滤波器转换为带阻滤波器
```
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

n = 10
z, p, k = signal.buttap(n)

b, a = signal.zpk2tf(z, p, k)
w, h = signal.freqs(b, a)

fl = 20
fh = 60

Wo = 2 * np.pi * np.sqrt(fl * fh)
Bw = 2 * np.pi * (fh - fl)

bt, at = signal.lp2bs(b, a, Wo, Bw)

w, h = signal.freqs(bt, at)

```

### LLM 生成 ： 调用程序实现将10阶低通巴特沃斯模拟滤波器转换为带阻滤波器并展示出

```
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

n = 10
z, p, k = signal.buttap(n)

b, a = signal.zpk2tf(z, p, k)
w, h = signal.freqs(b, a)

plt.figure()
plt.plot(w, 20 * np.log10(np.abs(h)))
plt.title('Lowpass Filter Frequency Response')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()

plt.figure()
plt.plot(w, np.angle(h, deg=False))
plt.title('Lowpass Filter Phase Response')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.show()

fl = 20
fh = 60

Wo = 2 * np.pi * np.sqrt(fl * fh)
Bw = 2 * np.pi * (fh - fl)

bt, at = signal.lp2bs(b, a, Wo, Bw)

w, h = signal.freqs(bt, at)

plt.figure()
plt.plot(w, 20 * np.log10(np.abs(h)))
plt.title('Bandstop Filter Frequency Response')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()

plt.figure()
plt.plot(w, np.angle(h, deg=False))
plt.title('Bandstop Filter Phase Response')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.show()
```

