# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：fircls1 

函数来源：[MATLAB fircls1]([[Constrained-least-squares linear-phase FIR lowpass and highpass filter design - MATLAB fircls1 - MathWorks 中国](https://ww2.mathworks.cn/help/signal/ref/fircls1.html)))

### 语法

```
b = fircls1(n,wo,dp,ds)
b = fircls1(n,wo,dp,ds,'high')
b = fircls1(n,wo,dp,ds,wt)
b = fircls1(n,wo,dp,ds,wt,'high')
b = fircls1(n,wo,dp,ds,wp,ws,k)
b = fircls1(n,wo,dp,ds,wp,ws,k,'high')
b = fircls1(n,wo,dp,ds,...,'design_flag')
```

### 说明

b = fircls1（n，wo，dp，ds） 生成 一个低通FIR滤波器，其中是 滤波器长度是归一化截止值 频率在 0 到 1 之间（其中 1 对应 到奈奎斯特频率），是最大通带 偏离 1（通带纹波），为 最大阻带偏差为 0（阻带纹波）。bn+1wodpds
b = fircls1（n，wo，dp，ds，'high'） 生成 高通 FIR 滤波器。 总是 对高通配置使用偶数滤波器顺序。这是 因为对于奇数阶，奈奎斯特频率的频率响应 必然是 0。如果指定奇值 ，则增量 它由 1.bfircls1nfircls1
b = fircls1（n，wo，dp，ds，wt） 和
b = fircls1（n，wo，dp，ds，wt，'high'） 指定 高于该频率（对于>）或更低的频率 其中 （对于< ） 滤波器保证满足给定的频段标准。这将 帮助您设计满足通带或阻带边缘要求的滤波器。 有四种情况：wtwtwowtwo
低通：
0 < wt < wo < 1：滤波器的幅度在 1 以内，频率范围< ω < 。dp0wt
0 < wo < wt < 1：滤波器的幅度在 0 以内，频率范围< ω < 。dswt1
高通：
0 < wt < wo < 1：滤波器的幅度在 0 以内，频率范围< ω < 。ds0wt
0 < wo < wt < 1：滤波器的幅度在 1 以内，频率范围< ω < 。dpwt1
b = fircls1（n，wo，dp，ds，wp，ws，k） 生成 具有加权功能的低通FIR滤波器， 其中 是过滤器长度，是 归一化截止频率是最大值 通带偏差为 1（通带纹波），为 与 0 的最大阻带偏差（阻带纹波）。
b = fircls1（n，wo，dp，ds，wp，ws，k，'高'） 生成 具有加权功能的高通FIR滤波器， 其中 < < .bwswowp
b = fircls1（n，wo，dp，ds,...,'design_flag'） 使 您可以监控滤波器设计，design_flag可以 是''
'trace'，用于文本显示 设计中使用的设计台
'plots'，对于滤波器的幅度、群延迟、 以及零点和极点。所有绘图在每个迭代步骤中都会更新。O's 开启 图是新迭代的估计极值，X 是 估计上一次迭代的极值，其中极值是峰值 （最大值和最小值）滤波。只有具有 相应的 O 和 X 相等。
'both'，用于文本显示和情节。

### 算法

`fircls1`使用迭代最小二乘算法 以获得等纹波响应。该算法是多重交换 使用拉格朗日乘数和 Kuhn-Tucker 条件的算法 在每次迭代中。

## Python函数描述：fircls1

函数来源：自定义

## Prompt 1 ： 生成 Python fircls1对应示例

参考下面MATLAB代码的envelope函数

```
n = 55;
wo = 0.3;
dp = 0.02;
ds = 0.008;
b = fircls1(n,wo,dp,ds,'both');
```

请使用python语言完成fircls函数对应示例的编写。

### LLM 生成 ： Python fircls1 函数示例

```
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 设计滤波器的参数
n = 150  # 滤波器阶数
f = [0, 0.4, 0.4, 1]  # 频率点，需要指定通带和阻带的边界
a = [1, 1, 0, 0]  # 在各个频率点的理想幅度响应
w = [1/0.02, 1/0.01]  # 权重因子，与上下界的倒数成比例

# 设计滤波器
b = signal.firls(n+1, f, a, weight=w)

# 计算和显示滤波器的频率响应
w, h = signal.freqz(b, worN=8000)

# 绘制总体频率响应
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(w/np.pi, np.abs(h), 'b')
plt.title('Digital filter frequency response')
plt.ylabel('Amplitude Response', color='b')
plt.grid()
plt.axis([0, 1, 0, 1.2])

# 绘制通带放大视图
plt.subplot(3, 1, 2)
plt.plot(w/np.pi, np.abs(h), 'b')
plt.ylabel('Amplitude Response (Passband)', color='b')
plt.grid()
plt.axis([0, 0.5, 0.98, 1.02])

# 绘制阻带放大视图
plt.subplot(3, 1, 3)
plt.plot(w/np.pi, np.abs(h), 'b')
plt.ylabel('Amplitude Response (Stopband)', color='b')
plt.xlabel('Frequency [rad/sample]')
plt.grid()
plt.axis([0.5, 1, -0.01, 0.01])

plt.tight_layout()
plt.show()
```