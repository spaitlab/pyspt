# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：sawtooth 

函数来源：[MATLAB sawtooth](https://ww2.mathworks.cn/help/signal/ref/sawtooth.html)

### 语法

x = sawtooth(t)  
x = sawtooth(t,xmax)

### 说明

x = sawtooth(t) 为时间数组 t 的元素生成周期为 2π 的锯齿波。sawtooth 类似于正弦函数，但会创建峰值为 -1 和 1 的锯齿波。锯齿波定义为在 2π 的倍数处为 -1，而在所有其他时间处以斜率为 1/π 随时间呈现线性增加。

### 输入参数

t — 时间数组
向量 | 矩阵 | N 维数组
时间数组，指定为向量、矩阵或 N 维数组。sawtooth 沿大小大于 1 的数组 t 的第一个维度进行运算。
数据类型: double

xmax — 波的最大值位置
1 (默认) | 介于 0 与 1 之间的标量
波的最大值位置，指定为 0 与 1 之间的标量。xmax 确定波达到最大值时介于 0 与 2π 之间的点。该函数在 0 到 2π×xmax 区间上从 -1 增加到 1，然后在 2π×xmax 到 2π 区间上从 1 线性减少到 -1。然后，该形状以 2π 为周期进行重复。
示例: xmax = 0.5 指定标准三角波，关于时间 π 对称，峰间振幅为 1。
数据类型: double

### 输出参量

x — 锯齿波
向量 | 矩阵 | N 维数组
锯齿波，以向量、矩阵或 N 维数组形式返回。



## Python函数描述：sawtooth

函数来源：[Python sawtooth](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html)

### 语法

scipy.signal.sawtooth(t, width=1)

### 说明

返回周期性的锯齿波或三角波形。
锯齿波形有一个2pi的周期,在0到width2pi的区间内从-1上升到1,然后在width2pi到2pi的区间内从1下降到-1。width必须在[0, 1]区间内。
注意,这不是带限的。它产生无数的谐波,这些谐波在频谱范围内来回反复。

### 输入参数

t — 数组类型
时间。

width — 可选的数组类型 
作为整个周期的一部分,上升斜坡的宽度。默认为1,产生上升斜坡,而0产生下降斜坡。width = 0.5产生三角波。如果是数组,会导致波形随时间变化,必须与t的长度相同。

### 输出参量

y — ndarray
包含锯齿波形的输出数组。


## Prompt 1 ： 请你根据该MATLAB生成锯齿波的示例编写相同功能的Python代码，python生成锯齿波的函数为：scipy.signal.sawtooth(t, width=1)

```
T = 10*(1/50);

fs = 1000;
t = 0:1/fs:T-1/fs;

x = sawtooth(2*pi*50*t);

plot(t,x)
grid on
```

### LLM 生成 ： 
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth

T = 10*(1/50)
fs = 1000
t = np.arange(0, T, 1/fs)

x = sawtooth(2*np.pi*50*t)

plt.plot(t, x)
plt.grid(True)
plt.show()
```

## Prompt 2 ： 请你根据该MATLAB生成三角波的示例编写相同功能的Python代码，python生成三角波的函数为：scipy.signal.sawtooth(t, width=1)

```
T = 10*(1/50);

fs = 1000;
t = 0:1/fs:T-1/fs;

x = sawtooth(2*pi*50*t,1/2);

plot(t,x)
grid on
```

### LLM 生成 ： 
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth

T = 10*(1/50)
fs = 1000
t = np.arange(0, T, 1/fs)

x = sawtooth(2*np.pi*50*t, width=0.5)

plt.plot(t, x)
plt.grid(True)
plt.show()
```




