# 信号处理仿真与应用 - 信号生成和预处理 - 波形生成

## MATLAB函数描述：square

函数来源：[MATLAB square](https://ww2.mathworks.cn/help/signal/ref/square.html)

### 语法
x = square(t)
x = square(t,duty)

### 说明
x = square(t) 为时间数组 t 的元素生成周期为 2π 的方波。square 类似于正弦函数，但会创建值为 -1 和 1 的方波。
x = square(t,duty) 生成指定占空比为 duty 的方波。占空比是方波为正的信号周期的百分比。

### 输入参数

t — 时间数组
向量 | 矩阵 | N 维数组
时间数组，指定为向量、矩阵或 N 维数组。square 沿大小大于 1 的数组 t 的第一个维度进行运算。
数据类型: single | double

duty — 占空比
50 (默认) | 从 0 到 100 的实数标量
占空比，指定为从 0 到 100 的实数标量。
数据类型: single | double

### 输出参量

x — 方波
向量 | 矩阵 | N 维数组
方波，以向量、矩阵或 N 维数组形式返回。



## Python函数描述：square

函数来源：自定义

### 包络函数定义：

import numpy as np

def square(t, duty=50):
    """
    Generate a square wave signal.
    
    Parameters:
    t : array_like
        Time vector.
    duty : float, optional
        Duty cycle percentage. Default is 50%.
    
    Returns:
    s : ndarray
        Square wave signal array. The signal is 1 for the high state
        and -1 for the low state within one period of 2*pi.
    """
    # Normalize t to (0, 2*pi)
    tmp = np.mod(t, 2 * np.pi)
    
    # Compute the normalized frequency for the duty cycle
    w0 = (2 * np.pi * duty) / 100
    
    # Assign 1 to the normalized t where it's less than w0, -1 elsewhere
    s = np.where(tmp < w0, 1, -1)
    
    return s

### 函数描述
这个 square 函数用于生成一个周期性方波信号，其基本形状是在一个周期内，信号在两个固定幅度之间快速切换。方波信号在电子和信号处理领域中非常有用，因为它们可以用于生成测试信号或模拟数字信号。

### 来源
该函数是模仿 MATLAB 中的 square 函数的行为而编写的，用于在 Python 中生成类似的方波信号。

### 参数
-  t (array_like): 时间向量，表示信号在不同时间点的采样。
-  duty (float, optional): 占空比百分比，默认值为50%。占空比是指在一个周期内信号保持在一个状态（通常是高状态）的时间比例。

### 返回值
-  s (ndarray): 返回一个 NumPy 数组，表示生成的方波信号。方波的高电平为1，低电平为-1。

###  注意事项
-  输入的时间向量 t 必须能够被转换成浮点数类型的 NumPy 数组。
-  占空比 duty 必须是一个 0 到 100 之间的浮点数。

###  函数工作原理
1. 使用 np.mod 将时间向量 t 归一化到 (0, 2*pi) 的范围内。
2. 根据占空比 duty 计算方波的高电平阈值 w0。
3. 使用 np.where 生成方波信号，当 tmp < w0 时为1，否则为-1。

### 使用场景
-  生成具有特定占空比的测试信号。
-  在数字电路设计中模拟时钟信号。
-  作为其他复杂信号生成算法的组成部分，如调幅（AM）信号。

### 改进建议
-  可以考虑添加对输入参数的验证，确保 t 和 duty 在合理的范围内。
-  如果需要更高级的方波生成功能，如可变频率或相位偏移，可以扩展此函数以接受这些额外的参数。
-  为了提高代码的可读性和可维护性，可以添加更多的注释和文档字符串来描述函数的行为。
-  如果函数将被用于实时信号生成，考虑优化性能，比如通过预先计算时间向量的相关属性来减少计算量。



## Prompt 1 ： 生成 Python square 函数

参考下面MATLAB代码的square函数
```
% 创建一个由 0 到 3π 之间的 100 个等间距数字组成的向量。生成周期为 2π 的方波。
t = linspace(0,3*pi)';
x = square(t);
plot(t/pi,x,'.-',t/pi,sin(t))
xlabel('t / \pi')
grid on

把它们转换为python代码
```
import numpy as np
import matplotlib.pyplot as plt

# 定义方波生成函数
def square(t, duty=50):
    # Normalize t to (0, 2*pi)
    tmp = np.mod(t, 2 * np.pi)
    
    # Compute the normalized frequency for the duty cycle
    w0 = (2 * np.pi * duty) / 100
    
    # Assign 1 to the normalized t where it's less than w0, -1 elsewhere
    s = np.where(tmp < w0, 1, -1)
    
    return s

# 创建时间向量 t，从 0 到 3*pi，共 100 个点
t = np.linspace(0, 3 * np.pi, 100)

# 生成方波信号
x = square(t)

# 绘制方波信号和对应的正弦波信号
plt.figure(figsize=(10, 5))
plt.plot(t / np.pi, x, '.-', label='Square Wave')
plt.plot(t / np.pi, np.sin(t), label='Sine Wave', color='r')

# 设置图的标题和坐标轴标签
plt.title('Square Wave and Sine Wave')
plt.xlabel('t / π')
plt.ylabel('Amplitude')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图形
plt.show()
```
请根据以上代码，完成Python语言square函数的编写，便于类似应用调用


### LLM 生成 ： Python square 函数

这是函数定义
def square(t, duty=50):
    """
    Generate a square wave with a period of 2*pi based on a given time vector t.
    
    Parameters:
    t : array_like
        Time vector.
    duty : float, optional
        Duty cycle percentage. Default is 50%.
    
    Returns:
    s : ndarray
        An array representing the square wave signal.
    """
    # Normalize t to the range [0, 2*pi)
    tmp = np.mod(t, 2 * np.pi)
    
    # Compute the threshold for the duty cycle
    w0 = (2 * np.pi * duty) / 100
    
    # Create the square wave signal using np.where
    s = np.where(tmp < w0, 1, -1)
    
    return s



## Prompt 2 ： 应用生成的 Python square 函数做MATLAB示例对应

这是原始信号x
```
import numpy as np
import matplotlib.pyplot as plt

# 创建时间向量 t，从 0 到 3*pi，共 100 个点
t = np.linspace(0, 3 * np.pi, 100)

# 生成方波信号
x = square(t)

这是函数定义
def square(t, duty=50):
    """
    Generate a square wave with a period of 2*pi based on a given time vector t.
    
    Parameters:
    t : array_like
        Time vector.
    duty : float, optional
        Duty cycle percentage. Default is 50%.
    
    Returns:
    s : ndarray
        An array representing the square wave signal.
    """
    # Normalize t to the range [0, 2*pi)
    tmp = np.mod(t, 2 * np.pi)
    
    # Compute the threshold for the duty cycle
    w0 = (2 * np.pi * duty) / 100
    
    # Create the square wave signal using np.where
    s = np.where(tmp < w0, 1, -1)
    
    return s
请给出应用函数square对原始信号x求包络线，并绘制图形，给出Python程序。

### LLM 生成 ： 
```
import numpy as np
import matplotlib.pyplot as plt

# 定义方波生成函数
def square(t, duty=50):
    """
    Generate a square wave with a period of 2*pi based on a given time vector t.
    
    Parameters:
    t : array_like
        Time vector.
    duty : float, optional
        Duty cycle percentage. Default is 50%.
    
    Returns:
    s : ndarray
        An array representing the square wave signal.
    """
    tmp = np.mod(t, 2 * np.pi)  # Normalize t to the range [0, 2*pi)
    w0 = (2 * np.pi * duty) / 100  # Compute the threshold for the duty cycle
    s = np.where(tmp < w0, 1, -1)  # Create the square wave signal using np.where
    return s

# 创建时间向量 t，从 0 到 3*pi，共 100 个点
t = np.linspace(0, 3 * np.pi, 100)

# 生成方波信号
x = square(t)

# 绘制方波信号
plt.figure(figsize=(10, 5))
plt.plot(t / np.pi, x, '.-', label='Square Wave')  # Plot the square wave

# 设置图的标题和坐标轴标签
plt.title('Square Wave Signal')
plt.xlabel('t / π')
plt.ylabel('Amplitude')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图形
plt.show()
```

## Prompt 3 ： 应用生成的 Python square.py 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：

**Matlab代码**
```
% 创建一个由 0 到 3π 之间的 100 个等间距数字组成的向量。生成周期为 2π 的方波。
t = linspace(0,3*pi)';
x = square(t);
plot(t/pi,x,'.-',t/pi,sin(t))
xlabel('t / \pi')
grid on
```

这是Python envelope.py
```
import numpy as np
import matplotlib.pyplot as plt

def square(t, duty=50):
    """
    Generate a square wave with a period of 2*pi based on a given time vector t.
    
    Parameters:
    t : array_like
        Time vector.
    duty : float, optional
        Duty cycle percentage. Default is 50%.
    
    Returns:
    s : ndarray
        An array representing the square wave signal.
    """
    tmp = np.mod(t, 2 * np.pi)  # Normalize t to the range [0, 2*pi)
    w0 = (2 * np.pi * duty) / 100  # Compute the threshold for the duty cycle
    s = np.where(tmp < w0, 1, -1)  # Create the square wave signal using np.where
    return s
```


### LLM 生成 ：

下面这是调用程序
```
import numpy as np
import matplotlib.pyplot as plt
from square import square

# 创建时间向量 t，从 0 到 3*pi，共 100 个点
t = np.linspace(0, 3 * np.pi, 100)
    
# 生成方波信号
x = square(t)
    
# 绘制方波信号和对应的正弦波信号
plt.figure(figsize=(10, 5))
plt.plot(t / np.pi, x, '.-', label='Square Wave')  # Plot the square wave
plt.plot(t / np.pi, np.sin(t), label='Sine Wave')  # Plot the sine wave
    
# 设置图的标题和坐标轴标签
plt.title('Square Wave and Sine Wave')
plt.xlabel('t / π')
plt.ylabel('Amplitude')
   
# 显示图例
plt.legend()
    
# 显示网格
plt.grid(True)
    
# 显示图形
plt.show()
``` 