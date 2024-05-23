# 信号处理仿真与应用-数字和模拟滤波器-线性系统变换

## MATLAB函数描述：tf2ss 

函数来源：[MATLAB tf2ss](https://ww2.mathworks.cn/help/signal/ref/tf2ss.html)

### 语法

[A,B,C,D\] = tf2ss(b,a)

### 说明

 [A, B, C, D] = tf2ss(b, a) 将连续时间或离散时间的单输入传递函数转换为等效的状态空间表示。

### 输入参数

b — 传递函数的分子系数
向量 | 矩阵
传递函数的分子系数，指定为向量或矩阵。如果 b 是一个矩阵，则 b 的每一行对应于系统的一个输出。

对于离散时间系统，b 包含 z 的降幂系数。

对于连续时间系统，b 包含 s 的降幂系数。

对于离散时间系统，b 必须具有与 a 的长度相等的列数。如果这两个数字不相等，请通过填充零使它们相等。您可以使用函数 eqtflength 来实现这一点。

a — 传递函数的分母系数
向量
传递函数的分母系数，指定为向量。

对于离散时间系统，a 包含 z 的降幂系数。

对于连续时间系统，a 包含 s 的降幂系数。

### 输出参量

A — 状态矩阵
矩阵
状态矩阵，返回为矩阵。如果系统由 n 个状态变量描述，则 A 为 n × n。

数据类型： single | double

B — 输入到状态矩阵
矩阵
输入到状态矩阵，返回为矩阵。如果系统由 n 个状态变量描述，则 B 为 n × 1。

数据类型： single | double

C — 状态到输出矩阵
矩阵
状态到输出矩阵，返回为矩阵。如果系统有 q 个输出并且由 n 个状态变量描述，则 C 为 q × n。

数据类型： single | double

D — 透传矩阵
矩阵
透传矩阵，返回为矩阵。如果系统有 q 个输出，则 D 为 q × 1。

数据类型： single | double

### 更多

传递函数
tf2ss 将给定系统的传递函数表示的参数转换为等效状态空间表示的参数。
对于离散时间系统，状态空间矩阵将状态向量 x、输入 u 和输出 y 关联起来：
$$
x(k+1)=Ax(k)+Bu(k)
$$
$$
y(k)=Cx(k)+Du(k)
$$

传递函数是系统冲激响应的 Z 变换。它可以用状态空间矩阵表示为
$$
H(z)=C(zI-A)^- B+D
$$
对于连续时间系统，状态空间矩阵将状态向量 x、输入 u 和输出 y 关联起来：
$$
x=Ax+Bu
$$
$$
y=Cx+Du
$$

传递函数是系统冲激响应的 Laplace 变换。它可以用状态空间矩阵表示为
$$
H(s)=B(s)/A(s)=(b_1⋅s^{n-1}+⋯+b_{n-1}⋅s+b_n)/(a_1⋅s^{m-1}+⋯+a_{m-1}⋅s+a_m )
$$



## python函数描述：2tf2ss

[python scipy.signal.tf2ss ](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2ss.html)

scipy.signal.tf2ss(num, den)传递函数转换为状态空间表示。

### 输入参数：

num, den：array_like
表示分子和分母多项式的系数序列，按降序排列。分母至少需要和分子一样长。

### 输出参量：

A, B, C, D：ndarray
系统的状态空间表示，处于控制器标准形式。

## Prompt 1 做MATLAB示例对应之将传递函数转换为状态空间形式

将传递函数转换为状态空间形式，考虑由传递函数描述的系统
$$
H(s)=[\begin{matrix}2s+3\\s^2+2s+1\end{matrix}]/(s^2+0.4s+1)
$$
使用 tf2ss 将其转换为状态空间形式。

参考下面Matlab代码，给出Python实现代码：

```matlab
b = [0 2 3; 1 2 1];
a = [1 0.4 1];
[A,B,C,D] = tf2ss(b,a)
```

对应的python代码为：

```python
import numpy as np
from scipy import signal

# 传递函数的分子和分母系数
b = [[0, 2, 3], [1, 2, 1]]
a = [1 , 0.4 , 1]
# 将传递函数转换为状态空间表示
A, B, C, D = signal.tf2ss(b, a)
```

## Prompt 2 ： 做MATLAB示例对应之质量-弹簧系统

一个一维离散时间振荡系统由单位质量 m 和单位弹性常数的弹簧连接到墙上。一个传感器以 Fs=5 Hz 采样质量的加速度 a。生成50个时间样本。定义采样间隔 Δt=1/Fs。

系统的传递函数有解析表达式：
$$
H(z)=(1-z^{-1}(1+cosΔt)+z^{-2}cosΔt)/(1-2z^{-1}cosΔt+z^{-2})
$$
系统受到正向单位冲激的激励。使用传递函数计算系统的时间演化。绘制响应。

参考下面Matlab代码，给出Python实现代码：

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
[A,B,C,D] = tf2ss(bf,af);

x = [0;0];
for k = 1:N
    y(k) = C*x + D*u(k);
    x = A*x + B*u(k);
end
hold on
stem(t,y,'*')
hold off
legend('tf','ss')
```

对应的python代码为：

```python
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

Fs = 5
dt = 1 / Fs
N = 50
t = np.arange(0, N) * dt
u = np.concatenate(([1], np.zeros(N - 1)))
bf = [1, -(1 + np.cos(dt)), np.cos(dt)]
af = [1, -2 * np.cos(dt), 1]
yf = signal.lfilter(bf, af, u)

plt.plot(t, yf, 'ob-')
plt.plot()
plt.xlabel('t')

A, B, C, D = signal.tf2ss(bf, af)
B = np.atleast_2d(B)
C = np.atleast_2d(C)

x = np.zeros((2, 1))
y = np.zeros(N)
for k in range(N):
    y[k] = np.dot(C, x) + D * u[k]
    x = np.dot(A, x) + B * u[k]

plt.stem(t, y, linefmt='r-', markerfmt='r*', basefmt='') # 添加了空字符串作为基线格式
plt.legend(['tf', 'ss'])
plt.show()
```
