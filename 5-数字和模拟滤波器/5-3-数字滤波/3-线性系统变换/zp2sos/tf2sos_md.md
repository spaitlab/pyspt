# 信号处理仿真与应用 - 数字滤波 - 线性系统变化

## MATLAB函数描述：zp2sos

函数来源：[MATLAB zp2sos](https://ww2.mathworks.cn/help/signal/ref/zp2sos.html?s_tid=doc_ta)

### 语法
[sos,g] = zp2sos(z,p,k)
[sos,g] = zp2sos(z,p,k,order)
[sos,g] = zp2sos(z,p,k,order,scale)
[sos,g] = zp2sos(z,p,k,order,scale,zeroflag)
sos = zp2sos(___)
### 输入参数
z— 零点
系统的零点，指定为向量。零必须是真实的，否则就会进来 复杂的共轭对。

例：[1 (1+1j)/2 (1-1j)/2]'

数据类型：
复数支持：是double

p— 极点
向量
系统的极点，指定为向量。杆子必须是真实的，否则就会进来 复杂的共轭对。

例：[1 (1+1j)/2 (1-1j)/2]'

数据类型：
复数支持：是double

k— 标量增益
标量
系统的标量增益，指定为标量。

数据类型：double

order— 行顺序
为“向上”（默认） |“向下”
行顺序，指定为以下项之一：

'up'— 对部分进行排序，使 sos 的第一行包含离单位最远的极点 圈。

'down'— 对截面进行排序，使第一行包含最接近单位圆的极点。sos

数据类型：char

scale— 增益和分子系数
的缩放 'none' （默认值） |'inf' |“二”
增益和分子系数的缩放，指定为以下值之一：

'none'— 不应用缩放。

'inf'— 应用无穷大范数缩放。

'two'— 应用 2 范数缩放。

将无穷范数缩放与 -ordering 一起使用可最大程度地减少 实现中溢出的概率。使用带有排序的 2 范数缩放可最大限度地降低峰值舍入噪声。'up''down'

注意

无穷范数和 2 范数缩放仅适用于直接形式 II 实现。

数据类型：char

zeroflag— 实零
的排序 false （默认） |真
实数零的排序，这些零点是彼此的负数，指定为逻辑标量。

如果指定为 ，则 函数根据与极点的接近程度对这些零进行排序。zeroflagfalse

如果指定为 ，则 函数将这些零放在一起。此选项生成的分子具有 中间系数等于零。zeroflagtrue

数据类型：logical

### 输出参量
sos— 二阶截面表示
矩阵
二阶截面表示，以矩阵形式返回。 是 L×6 矩阵sos，如果传递函数有 n 个零点和 m 个极点，则 L 是最接近的大整数 大于或等于 max（n/2，m/2）

g— 整体系统增益
实标量
整体系统增益，以实际标量形式返回。

## Python函数描述：tf2sos

函数来源：control库中的tf2zpk函数

### 函数定义：

    # 调用scipy中的zpk2sos函数
    import numpy as np
    from scipy.signal import zpk2sos
    sos = signal.zpk2sos(z, p, k, pairing='nearest')


### 参数
z：一个包含零点的一维数组。
p：一个包含极点的一维数组。
k：一个常数，表示系统的增益。
pairing：可选参数，表示如何匹配零点和极点。可以选择'nearest'（默认）或者其他选项。

### 返回值
sos：二阶级联型滤波器的表示形式
### 使用场景
这个函数可以将零点-极点-增益（ZPK）表示的滤波器转换为二阶级联型（Second-Order Sections，SOS）表示，从而方便进行数字信号处理

## Prompt 1 ： 使用输出以零极点增益形式表示的函数设计一个五阶巴特沃斯低通滤波器。将截止频率指定为奈奎斯特频率的五分之一。将结果转换为二阶截面。可视化频率响应

参考下面MATLAB代码的zpk2sos函数
```
[z,p,k] = butter(5,0.2);
sos = zp2sos(z,p,k)
freqz(sos)
```

我们采用Python语言实现
```
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 设计五阶巴特沃斯低通滤波器，指定截止频率为奈奎斯特频率的五分之一
order = 5
nyquist_freq = 0.5
cutoff_freq = nyquist_freq / 5
b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)

# 将滤波器表示为零极点增益形式
z, p, k = signal.tf2zpk(b, a)

# 将零极点形式转换为二阶截面
sos = signal.zpk2sos(z, p, k)

# 绘制频率响应
w, h = signal.sosfreqz(sos, worN=8000)
plt.figure()
plt.plot(w, 20 * np.log10(np.abs(h)))
plt.title('Frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.show()
```




