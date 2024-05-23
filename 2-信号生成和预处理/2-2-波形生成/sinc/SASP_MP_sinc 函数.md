# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：sinc 函数 

函数来源：[MATLAB sinc 函数](https://ww2.mathworks.cn/help/signal/ref/sinc.html)

### 语法

y = sinc(x)

### 说明

y = sinc(x) 返回数组 y，其元素是输入 x 的元素的 sinc。输出 y 与 x 的大小相同。

### 输入参数

x — 输入数组
标量值 | 向量 | 矩阵 | N 维数组
输入数组，指定为实数值或复数值标量、向量、矩阵或 N 维数组。当 x 为非标量时，sinc 是按元素运算。
数据类型: single | double
复数支持: 是

### 输出参量

y — 输入的 sinc
标量值 | 向量 | 矩阵 | N 维数组
输入数组 x 的 sinc，以与 x 大小相同的实数值或复数值标量、向量、矩阵或N 维数组形式返回。

## Python函数描述：sinc 函数 

函数来源：[Python sinc 函数](https://numpy.org/doc/stable/reference/generated/numpy.sinc.html)

### 语法

numpy.sinc(x)

### 说明

返回标准化sinc函数。
sinc函数等于sin(πx)/(πx)，对于任何参数x≠0。 sinc(0)取极限值1，使得sinc不仅在任何地方连续，而且是无限可微的。

### 输入参数

x — ndarray
用于计算sinc(x)的值的数组（可能是多维的）。

### 输出参量

out — ndarray
sinc(x)，具有与输入相同的形状。

## Prompt 1 ： 请你根据该MATLAB示例编写相同功能的Python代码，python的sinc函数为：numpy.sinc(x)
```
rng default

t = 1:10;
x = randn(size(t))';
ts = linspace(-5,15,600);
[Ts,T] = ndgrid(ts,t);
y = sinc(Ts - T)*x;

plot(t,x,'o',ts,y)
xlabel Time, ylabel Signal
legend('Sampled','Interpolated','Location','SouthWest')
legend boxoff
```

### LLM 生成

```
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
t = np.arange(1, 11)
x = np.random.randn(t.size)
ts = np.linspace(-5, 15, 600)

y = np.zeros_like(ts)
for i, ti in enumerate(t):
    y += np.sinc(ts - ti) * x[i]

plt.plot(t, x, 'o', label='Sampled')
plt.plot(ts, y, label='Interpolated')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend(loc='lower left')
plt.show()
```
