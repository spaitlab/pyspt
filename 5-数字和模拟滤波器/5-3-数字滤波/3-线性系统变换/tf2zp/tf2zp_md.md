# 信号处理仿真与应用 - 数字滤波 - 线性系统变化

## MATLAB函数描述：tf2zp

函数来源：[MATLAB tf2zp](https://ww2.mathworks.cn/help/signal/ref/tf2zp.html)

### 语法

[z,p,k] = tf2zp(b,a)

### 说明
。tf2zp函数将传递函数表示为b/a

传递给tf2zp函数的传递函数必须是连续系统，而非离散系统。如果传递给tf2zp的多项式系数不存在，则tf2zp函数会返回一个空的向量。
### 输入参数

b— 传递函数分子系数
向量 |矩阵
传递函数分子系数，指定为向量或矩阵。如果是一个矩阵，则每一行对应于系统的输出。 包含系数 在 S 的降序幂中。的列数必须小于或等于 a 的长度。bbbb

数据类型： |singledouble

a— 传递函数分母系数
向量
传递函数分母系数，指定为向量。 包含 s 的降序幂系数。a

数据类型： |singledouble
### 输出参量
z— 零点
矩阵
系统的零点，以矩阵形式返回。 包含 分子在其列中为零。 有多少列就有多少列 输出。zz

p— 极点
列向量
系统的极点，作为列向量返回。 包含 传递函数的分母系数的极点位置。p

k— 增益
列向量
系统的增益，以列向量的形式返回。 包含 每个分子传递函数的增益。k


## Python函数描述：tf2zp

函数来源：control库中的tf2zpk函数

### 函数定义
    import numpy as np
    from scipy import signal

    def tf2zp(num, den):
    # 调用scipy中的tf2zpk函数
    z, p, k = signal.tf2zpk(num, den)
    
    将复数根按照实部和虚部分开
    zeros_real = np.real(z)
    zeros_imag = np.imag(z)
    poles_real = np.real(p)
    poles_imag = np.imag(p)
    
    return zeros_real, zeros_imag, poles_real, poles_imag, k

### 参数

num — 传递函数的分子多项式系数
用向量表示 

den — 传递函数的分母多项式系数
用向量表示

### 返回值
返回其对应的零点（z）、极点（p）和增益（k）
### 注意事项
传递给tf2zp函数的传递函数必须是连续系统，而非离散系统。如果传递给tf2zp的多项式系数不存在，则tf2zp函数会返回一个空的向量。
### 函数工作原理
tf2zpk函数将传递函数表示为num/den，并返回其对应的零点（z）、极点（p）和增益（k）。
### 使用场景
实现传递函数到极点-零点-增益的转换

## Prompt 1 ： 生成 Python tf2zp 函数

参考下面MATLAB代码的tf2zp函数
```
b = [2 3];
a = [1 1/sqrt(2) 1/4];

[b,a] = eqtflength(b,a);
[z,p,k] = tf2zp(b,a)

```

我们采用Python语言实现
```
# 定义一个传递函数
num = [2,3]
den = [1, 1/sqrt(2), 1/4]

# 调用tf2zp函数获得零点和极点
zeros_real, zeros_imag, poles_real, poles_imag, k = tf2zp(num, den)

# 打印结果
print("传递函数的零点（实部）:", zeros_real)
print("传递函数的零点（虚部）:", zeros_imag)
print("传递函数的极点（实部）:", poles_real)
print("传递函数的极点（虚部）:", poles_imag)
print("传递函数的增益:", k)







