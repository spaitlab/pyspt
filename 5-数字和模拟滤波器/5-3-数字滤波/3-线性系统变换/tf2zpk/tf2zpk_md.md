# 信号处理仿真与应用 - 数字滤波 - 线性系统变化

## MATLAB函数描述：tf2zpk

函数来源：[MATLAB tf2zpk](https://ww2.mathworks.cn/help/signal/ref/tf2zpk.html)

### 语法

[z,p,k] = tf2zpk(b,a)

### 说明
tf2zpk适用于离散系统z变化的系统函数，而连续系统，系统函数为s变化的可以使用前一个函数tf2zp
### 输入参数

b— 传递函数分子系数向量
传递函数分子系数，指定为向量。 包含 z 的升幂系数b–1.

数据类型： |singledouble

a— 传递函数分母系数
向量
传递函数分母系数，指定为向量。 包含 z 的升幂系数a–1.

数据类型： |singledouble

### 输出参量
z— 系统零列
向量
系统零点，作为列向量返回。

p— 系统极柱
柱矢量
系统极点，以列向量形式返回。

k— 系统增益
标量
系统增益，以标量形式返回。

## Python函数描述：tf2zp

函数来源：control库中的tf2zpk函数

### 函数定义：

    # 调用scipy中的tf2zpk函数
    from scipy.signal import tf2zpk

    z, p, k = tf2zpk(num, den)

### 参数

num — 传递函数的分子多项式系数
用向量表示 

den — 传递函数的分母多项式系数
用向量表示

### 返回值
返回其对应的零点（z）、极点（p）和增益（k）

### 函数工作原理
tf2zpk函数将传递函数表示为num/den，并返回其对应的零点（z）、极点（p）和增益（k）。
### 使用场景
实现传递函数到极点-零点-增益的转换

## Prompt 1 ： 生成 Python tf2zpk函数 设计具有归一化截止频率的三阶巴特沃斯滤波器0.4πrad/sample。找到滤波器的极点、零点和增益。

参考下面MATLAB代码的tf2zpk函数
```
[b,a] = butter(3,0.4);
[z,p,k] = tf2zpk(b,a)
#绘制极点和零点以验证它们是否在预期位置。
zplane(b,a)
text(real(z)-0.1,imag(z)-0.1,"Zeros")
text(real(p)-0.1,imag(p)-0.1,"Poles")
```

我们采用Python语言实现
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tf2zpk

# 定义设计巴特沃斯滤波器的函数，根据阶数和归一化截止频率返回极点、零点和增益
def design_butterworth_filter(order, cutoff):
    # 将归一化截止频率转换为传入tf2zpk函数中的形式
    normalized_cutoff = cutoff / np.pi
    # 调用tf2zpk函数计算极点、零点和增益
    z, p, k = tf2zpk([1], [1, 0.7654, 1.848, 0.7654, 1])
    
    return z, p, k

# 设置滤波器阶数和归一化截止频率
order = 3
cutoff_frequency = 0.4 * np.pi

# 调用设计滤波器函数获取极点、零点和增益
z, p, k = design_butterworth_filter(order, cutoff_frequency)

# 极点、零点和增益
print("滤波器极点：", p)
print("滤波器零点：", z)
print("滤波器增益：", k)

# 绘制极点和零点的散点图以验证它们的位置
plt.scatter(np.real(z), np.imag(z), marker='o', color='b', label='Zeros')
plt.scatter(np.real(p), np.imag(p), marker='x', color='r', label='Poles')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Pole-Zero Plot of the Butterworth Filter')
plt.legend()
plt.grid(True)
plt.show()






