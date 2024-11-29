# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波器分析

## MATLAB函数描述：isstable

函数来源：[MATLAB isstable](https://ww2.mathworks.cn/help/signal/ref/isstable.html)

### 语法

flag = isstable(b,a)
flag = isstable(sos)
flag = isstable(d)

### 说明

flag = isstable(b,a)当输入分子系数b分母系数为a的系统函数稳定时，输出1，反之，输出0。
flag = isstable(sos)如果滤波器系数sos稳定则返回逻辑1，否则返回逻辑0。
flag = isstable(d)如果数字滤波器d稳定返回逻辑1，否则返回逻辑0。

### 输入参数

b,a — 传递函数参数
行向量
传递函数参数，由行向量的形式给出。传递函数由z^-1定义。
数据类型: single | double

sos — 直接Ⅱ型传递函数
6维矩阵
直接Ⅱ型传递函数，由6维矩阵给出。
数据类型: single | double

### 输出参量

flag — 逻辑输出
1/0
当输出为1时，表示输入为稳定系统。
传递函数参数，由行向量的形式给出。传递函数由z^-1定义。
数据类型: single | double


## Python函数描述：isatable

函数来源：自定义

### 函数定义：

def isstable(b, a):
    system = ctl.TransferFunction(b, a)
    poles = ctl.poles(system)
    return np.all(np.real(poles) < 0)


这段代码定义了一个`isstable`函数，你可以传入分子系数b和分母系数a，这个函数可以用于检查任何数字滤波器是否是稳定的。

### 参数
- `b,a`: 输入传递函数分子系数b和分母系数a。

### 注意事项
- 该函数只适用于输入传递函数系数。

### 函数工作原理
1. 使用 `ctl.TransferFunction` 、 `ctl.poles`求解传递函数极点。
2. 判断极点的实部是否小于零。

### 使用场景
这个函数可以用于检查任何数字滤波器是否是稳定的。

### 改进建议
- 可以添加对zpk模型输入稳定性判断。



## Prompt 1 ： 生成 Python isstable 函数

import control as ctl
import numpy as np


def isstable(b, a):
    system = ctl.TransferFunction(b, a)

    # 获取系统的极点
    poles = ctl.poles(system)

    # 判断系统是否稳定（所有极点的实部需要小于0）
    return np.all(np.real(poles) < 0)


# 示例使用

b = [1]  # 分子
a = [1, 1, -10]  # 分母
print("系统是否稳定？", isstable(b, a))

