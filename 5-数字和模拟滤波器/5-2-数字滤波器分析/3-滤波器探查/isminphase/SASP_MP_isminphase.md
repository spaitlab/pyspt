# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波器分析

## MATLAB函数描述：isminphase

函数来源：[MATLAB isminphase](https://ww2.mathworks.cn/help/signal/ref/isminphase.html)

### 语法

flag = isminphase(b,a)
flag = isminphase(sos)
flag = isminphase(d)
flag = isminphase(___,tol)

### 说明

flag = isminphase(b,a)如果输入分子系数b与分母系数a符合最小相位系统，则返回逻辑1，否则返回逻辑0。
flag = isminphase(sos)如果滤波器系数sos为最小相位系统则返回逻辑1，否则返回逻辑0。
flag = isminphase(d)如果数字滤波器d为最小相位系统则返回逻辑1，否则返回逻辑0。
flag = isminphase(___,tol)使用容差值tol判决最小相位系统。

### 输入参数

b,a — 传递函数参数
行向量
传递函数参数，由行向量的形式给出。传递函数由z^-1定义。
数据类型: single | double

sos — 直接Ⅱ型传递函数
6维矩阵
直接Ⅱ型传递函数，由6维矩阵给出。
数据类型: single | double

tol — 容限
正标量
当两个标量之差小于tol时，matlab认为两个标量相等。
数据类型: double

### 输出参量

flag — 逻辑输出
1/0
当输出为1时，表示输入为最小相位系统。
传递函数参数，由行向量的形式给出。传递函数由z^-1定义。
数据类型: single | double



## Python函数描述：isminphase

函数来源：自定义

### 函数定义：

def is_min_phase(b, a=1.0):
    z, p, k = tf2zpk(b, a)
    return np.all(np.abs(z) < 1) & np.all(np.abs(p) < 1)


这段代码定义了一个`isminphase`函数，你可以传入分子系数b和分母系数a，这个函数可以用于检查任何数字滤波器是否是最小相位的。

### 参数
- `b,a`: 输入传递函数分子系数b和分母系数a。

### 注意事项
- 该函数只适用于输入传递函数系数。

### 函数工作原理
1. 使用 `tf2zpk` 将传递函数模型转换为零极点模型。
2. 判断零点和极点是否在单位圆内。

### 使用场景
这个函数可以用于检查任何数字滤波器是否是最小相位的。

### 改进建议
- 可以添加对zpk模型输入的最小相位判断。



## Prompt 1 ： 生成 Python isminphase 函数

import numpy as np
from scipy.signal import tf2zpk


def is_min_phase(b, a=1.0):
    # Convert transfer function coefficients to zeros, poles, and gain
    z, p, k = tf2zpk(b, a)

    # Check if all zeros are inside the unit circle
    return np.all(np.abs(z) < 1) & np.all(np.abs(p) < 1)


# 示例使用
b = [1, -2, 2]  # Coefficients of the numerator
a = [1, 0.5]    # Coefficients of the denominator
result = is_min_phase(b, a)
print(np.abs(np.roots(b)))
print(np.abs(np.roots(a)))
print(f"The system is minimum phase: {result}")

