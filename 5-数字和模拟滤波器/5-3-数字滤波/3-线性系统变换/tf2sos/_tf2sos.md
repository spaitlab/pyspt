# 信号处理仿真与应用-数字和模拟滤波器-线性系统变换

## MATLAB函数描述：tf2sos 

函数来源：[MATLAB tf2sos ][Convert digital filter transfer function data to second-order sections form - MATLAB tf2sos - MathWorks 中国](https://ww2.mathworks.cn/help/signal/ref/tf2sos.html)

### 语法

[sos,g] = tf2sos(b,a)
[sos,g] = tf2sos(b,a,order)
[sos,g] = tf2sos(b,a,order,scale)
sos = tf2sos(___)

### 说明

[sos,g] = tf2sos(b,a) 函数用转移函数系数向量 b 和 a 表示的数字滤波器，找到等效于第二阶节形式的矩阵 sos，并返回增益 g。

[sos,g] = tf2sos(b,a,order) 指定 sos 中行的顺序。

[sos,g] = tf2sos(b,a,order,scale) 指定所有第二阶节的增益和分子系数的缩放。

sos = tf2sos(___) 将整体系统增益嵌入到第一个节中。

### 输入参数

b, a — 传递函数系数
向量
传递函数系数，指定为向量。使用 b 和 a 表示传递函数，如下所示：
$$
H(k)=B(z)/A(z)=(b_{1}+b_{2}z^{-1}+⋯+b_{n+1}z^{-n})/(a_{1}+a_{2}z^{-1}+⋯+a_{m+1}z^{-m})
$$
示例：b = [1 3 3 1]/6 和 a = [3 0 1 0]/3 指定了一个三阶巴特沃斯滤波器，其标准化 3 dB 频率为 0.5π 弧度/样本。

数据类型：double

order — 行顺序
'up'（默认）| 'down'
行顺序，指定为以下之一：

'up' — 对节进行排序，使得 sos 的第一行包含离单位圆最远的极点。

'down' — 对节进行排序，使得 sos 的第一行包含离单位圆最近的极点。

数据类型：char

scale — 增益和分子系数的缩放
'none'（默认）| 'inf' | 'two'
增益和分子系数的缩放，指定为以下之一：

'none' — 不应用缩放。

'inf' — 应用无穷范数缩放。

'two' — 应用 2-范数缩放。

使用 'up'-ordering 的无穷范数缩放可以最小化实现中的溢出概率。使用 'down'-ordering 的 2-范数缩放可以最小化峰值舍入误差。

注意：无穷范数和 2-范数缩放仅适用于直接形式 II 的实现。

数据类型：char

### 输出参量

sos — 第二阶段节表示
矩阵
第二阶段节表示，返回为矩阵。sos 是一个 L × 6 的矩阵：
$$
sos= \begin{matrix}
b01 & b11 & b21 & 1 & a11 & a21\\
b02 & b12 & b22 & 1 & a12 & a22 \\
 ⋮  &  ⋮   &  ⋮  & ⋮  &  ⋮  &  ⋮ \\
b0L & b1L & b2L & 1 &a1L & a2L\\
\end{matrix}
$$
其中每行包含 H(z) 的第二阶段节的分子和分母系数 bik 和 aik：
$$
H(k)=\sum_{k=0}^{L} H_k(z)=g\sum_{k=0}^{L}(b_{0k}+b_{1k}z^{-1}+b_{2k}z^{-2})/(a_{0k}+a_{1k}z^{-1}+a_{2k}z^{-2})
$$
g — 整体系统增益
实数标量
整体系统增益，返回为实数标量。

如果你用一个输出参数调用 tf2sos，函数将整体系统增益嵌入到第一节 H1(z) 中，如下所示：
$$
H(k)=\sum_{k=0}^{L} H_k(z)
$$
注意：在缩放直接形式 II 结构时，将增益嵌入到第一节中不推荐，可能导致不稳定的缩放。为了避免嵌入增益，请使用带有两个输出的 tf2sos。

### 算法

tf2sos 使用四步算法确定输入传递函数系统的第二阶段节表示：

1.它找到由 b 和 a 给定的系统的极点和零点。

2.它使用函数 zp2sos，该函数首先使用 cplxpair 函数将零点和极点分组成复共轭对。然后，zp2sos 根。据以下规则将极点和零点成对匹配形成第二阶段节：

a.将最接近单位圆的极点与最接近这些极点的零点匹配。

b.将次接近单位圆的极点与最接近这些极点的零点匹配。

c.继续，直到所有极点和零点都匹配。

3.tf2sos 将实极点分组到具有绝对值最接近它们的实极点的节中。对于实零点，同样的规则适用。

4.它根据极点对于单位圆的接近程度对节进行排序。tf2sos 通常将具有最接近单位圆的极点的节最后进行级联。您可以通过将 order 指定为 'down' 来告诉 tf2sos 以相反的顺序对节进行排序。

tf2sos 使用 scale 中指定的范数对节进行缩放。对于任意 H(ω)，缩放定义为：
$$
||H||_p=[1/2π\int_{0}^{2π} |H(w)|^p \, dw]^{1/p}
$$
其中 p 可以是 ∞ 或 2。算法遵循此缩放，试图最小化固定点滤波器实现中的溢出或峰值舍入误差。

## python函数描述：sos2tf 

函数来源：[python scipy.signal.sos2tf ](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sos2tf.html)

scipy.signal.sos2tf 从一系列二阶节中返回单个传递函数

### 输入参数 :

 sos ：array_like 

二阶滤波器系数的数组，必须具有形状 (n_sections, 6)。请参阅 sosfilt 以获取 SOS 滤波器格式规范。 

### 输出参量 : 

b：ndarray 分子多项式系数。 

a ：ndarray 分母多项式系数。

###  注意事项 

版本 0.16.0 中新增。 

## Prompt 1 ：根据MATLAB代码生成 Python 代码

设计一个Butterworth四阶低通滤波器，使用函数butter。将截止频率指定为Nyquist频率的一半。将滤波器实现为二阶节。通过比较它们的分子和分母来验证两种表示是否相同。

参考下面MATLAB代码：

```matlab
[nm,dn] = butter(4,0.5);
[ss,gn] = tf2sos(nm,dn);
numers = [conv(ss(1,1:3),ss(2,1:3))*gn;nm]
denoms = [conv(ss(1,4:6),ss(2,4:6));dn]
```

给出python代码：

```python
from scipy.signal import butter, sos2tf, convolve
import numpy as np

# 生成4阶低通Butterworth滤波器
sos = butter(4, 0.5, output='sos')
# 转换为数字滤波器的分子和分母多项式系数
b, a = sos2tf(sos)
numers = [np.convolve(sos[0, 0:3], sos[1, 0:3]) * sos[0, 3] * sos[1, 3], b]
denoms = [convolve(sos[0, 3:6], sos[1, 3:6]), a]
```

## Prompt 2 ：根据MATLAB代码生成 Python 代码

质量-弹簧系统

一维离散时间振荡系统由单位质量 m 连接到一个单位弹性常数的弹簧，固定在墙上。一个传感器以 Fs=5 Hz 采样质量的加速度 a。

生成 50 个时间样本。定义采样间隔 Δt=1/Fs。

系统的传递函数有解析表达式：
$$
H(z)=(1-z^{-1}(1+cosΔt)+z^{-2}cosΔt)/(1-2z^{-1}cosΔt+z^{-2})
$$
系统受到正向单位冲激的激励。使用传递函数计算系统的时间演化。绘制响应。

MATLAB代码为：

```matlab
Fs = 5;
dt = 1/Fs;
N = 50;
t = dt*(0:N-1);
u = [1 zeros(1,N-1)];
bf = [1 -(1+cos(dt)) cos(dt)];
af = [1 -2*cos(dt) 1];
yf = filter(bf,af,u);
stem(t,yf,'o')
xlabel('t')
sos = tf2sos(bf,af);
yt = sosfilt(sos,u);
stem(t,yt,'filled')
```

对于的python代码为：

```python
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import sosfilt,lfilter,tf2sos

# Define parameters

Fs = 5
dt = 1 / Fs
N = 50
t = np.arange(0, N) * dt
u = np.concatenate(([1], np.zeros(N - 1)))
bf = [1, -(1 + np.cos(dt)), np.cos(dt)]
af = [1, -2 * np.cos(dt), 1]
yf = lfilter(bf, af, u)

plt.plot(t, yf, 'ob-')
plt.plot()
plt.xlabel('t')

plt.show()

# 将滤波器系数转换为二阶节

sos = tf2sos(bf, af)

# 使用二阶节对输入信号进行滤波。

yt = sosfilt(sos, u)

plt.stem(t, yt, linefmt='C1-', markerfmt='o', basefmt='k-')
plt.xlabel('t')
plt.title('Filtered Output using SOS')
plt.show()
```

