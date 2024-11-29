# 信号处理仿真与应用-数字和模拟滤波器-线性系统变换

## MATLAB函数描述：ss2zp

函数来源：[MATLAB ss2zp](https://ww2.mathworks.cn/help/signal/ref/ss2zp.html)

### 语法

[z,p,k] = ss2zp(A,B,C,D)

[z,p,k] = ss2zp(A,B,C,D,ni)

### 说明

[z,p,k] = ss2zp(A,B,C,D) 转换状态空间表示
$$
x = Ax + Bu
$$

$$
y = Cx + Du
$$

将给定的连续时间或离散时间系统转换为等效的零极增益表示。
$$
H(s) =Z(s)/P(s)=k(s-z_1)(s-z_2)⋯(s-z_n)/(s-p_1)(s-p_2)⋯(s-p_n)
$$
其零点、极点和增益以因子形式表示传递函数。

[z,p,k] = ss2zp(A,B,C,D,ni)  表示系统有多个输入，第n个输入被单位脉冲激发。

### 输入参数

A — 状态矩阵

矩阵

状态矩阵。如果系统有r个输入和q个输出并且由n个状态变量描述，那么A是n × n的。

数据类型: single | double

B — 输入到状态矩阵

矩阵

输入到状态矩阵。如果系统有r个输入和q个输出并且由n个状态变量描述，那么B是n × r。

数据类型: single | double

C — 状态到输出矩阵

状态到输出矩阵。如果系统有r个输入和q个输出并且由n个状态变量描述，那么C就是q × n。

数据类型: single | double

D — 直接传递矩阵

矩阵

直接传递矩阵。如果系统有r个输入和q个输出并且由n个状态变量描述，那么D就是q × r。

数据类型: single | double

ni — 输入索引

1(默认)|整数标量

输入索引，指定为整数标量。如果系统有r个输入，可以使用ss2zp并在后面添加参数ni=1,⋯r,来计算对施加第n个输入上的单位冲击响应。指定这个参数会导致ss2zp使用B和D矩阵的第n列。

数据类型: single | double

### 输出参量

z — 零点

矩阵

系统的零点，以矩阵形式表示  。矩阵z的每一列包含系统的分子零点。z的列数等于输出个数（矩阵C的行数）。

p — 极点

列向量

系统的极点，以列向量的形式返回。向量p包含传递函数分母系数的极点位置。

k — 增益

列向量

系统的增益，以列向量的形式返回。向量k包含每个分子传递函数的增益。

### 算法

`ss2zp` 从 A 矩阵的特征值中找到极点。零点是广义特征值问题的有限解：

𝑧=eig([A B;C D],diag([ones(1,n) 0]));*z*=eig([A B;C D],diag([ones(1,n) 0]));

在许多情况下，此算法会产生虚假的大但有限的零点。`ss2zp` 将这些大零点解释为无穷。

`ss2zp` 通过求解第一个非零马尔科夫参数来找到增益。

## python函数描述：ss2zpk

函数来源：[python scipy.signal.ss2zpk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ss2zpk.html)

### 语法

scipy.signal.ss2zpk(A,B,C,D,input=0)

将状态空间表示转换为零极点增益表示。

其中，A，B，C，D定义了一个线性状态空间系统。该系统有p个输入，q个输出和n个状态变量。

### 参数

A：类数组

(n,n)形的状态矩阵（或系统矩阵）

B：类数组

(n,p)形的输入矩阵

C：类数组

(q,n)形的输出矩阵

D：类数组

(q,p)形的传递矩阵（或前馈矩阵）

input：int，optional

对于多输入系统，要使用的输入索引。

### 返回值

z，p：序列

零点和极点

k：浮点数

系统增益

## Prompt 1：离散时间系统的零点，极点和增益

考虑一个由传递函数定义的离散时间系统
$$
H(z)=(2+3z^{-1})/(1+0.4z^{-1}+z^{-2})
$$
直接从传递函数确定它的零点、极点个增益。用零点填充分子多项式，使其与分母多项式具有相同的长度。

将系统表达为状态空间形式，并使用 `ss2zp` 确定零点、极点和增益。

相应的MATLAB代码为：

```matlab
b = [2 3 0];
a = [1 0.4 1];
[z,p,k] = tf2zp(b,a)
[A,B,C,D] = tf2ss(b,a);
[z,p,k] = ss2zp(A,B,C,D,1)
```

对应的python代码：

```python
import scipy as scipy

b = [2, 3, 0]
a = [1, 0.4, 1]
# 使用 tf2zpk 函数计算传递函数的零极点
[z1, p1, k1] = scipy.signal.tf2zpk(b, a)
print("Transfer Function Zeros:", z1)
print("Transfer Function Poles:", p1)
print("Transfer Function Gain:", k1)
# 使用 tf2ss 函数将传递函数转换为状态空间表示
[A, B, C, D] = scipy.signal.tf2ss(b, a)
# 使用 ss2zpk 函数计算状态空间表示的零极点
[z, p, k] = scipy.signal.ss2zpk(A, B, C, D)
print("State Space Zeros:", z)
print("State Space Poles:", p)
print("State Space Gain:", k)
```





