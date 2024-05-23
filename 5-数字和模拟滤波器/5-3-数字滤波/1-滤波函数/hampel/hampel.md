# 信号处理仿真与应用 - 数字滤波

## MATLAB函数描述：hampel
 

函数来源：[MATLAB hampel](https://ww2.mathworks.cn/help/signal/ref/hampel.html)

### 语法

y = hampel(x)
y = hampel(x,k)
y = hampel(x,k,nsigma)
[y,j] = hampel(___)
[y,j,xmedian,xsigma] = hampel(___)
hampel(___)

### 说明

y = hampel(x) 对输入向量 x 应用汉普尔滤波器，以检测和去除异常值。对于 x 的每个样本，该函数都会计算由该样本及其周围六个样本（每边三个）组成的窗口的中值。它还会使用中位数绝对偏差估算每个样本与其窗口中位数的标准偏差。如果样本与中位数相差超过三个标准差，就会被中位数取代。如果 x 是一个矩阵，那么函数会将 x 的每一列视为一个独立通道。
y = hampel(x,k)指定测量窗口中每个 x 样本两侧的邻居数量 k。
y = hampel(x,k,nsigma)指定 x 样本与本地中位数必须相差多少个标准差 nsigma，才能用中位数替换。
[y,j] = hampel(___) 也会返回一个逻辑矩阵，该矩阵在所有被识别为离群值的点的位置上都为真。此语法接受前面语法中的任何输入参数。
[y,j,xmedian,xsigma] = hampel(____)也会返回每个 x 元素的局部中值和估计标准偏差。
hampel(___) 在没有输出参数的情况下，绘制出滤波后的信号图，并标注出去除的异常值。

### 输入参数

x - 输入信号
向量 | 矩阵
输入信号，指定为向量或矩阵。如果 x 是矩阵，则 hampel 会将 x 的每一列视为一个独立通道。
例如：cos(pi/4*(0:159))+randn(1,160)是单通道行矢量信号。
例：cos(pi./[4;2]*(0:159))'+randn(160,2) 是双通道信号。
数据类型：single | double

k - 两侧邻居的数量
3（默认值） | 整数标量
样本 xs 两侧的相邻样本数，以整数标量形式指定。靠近信号边缘的样本，如果一侧的样本少于 k，就会与较小窗口的中值进行比较。
数据类型：single

nsigma - 标准差个数
3（默认值） | 实数标量
x 样本与本地中位数必须相差多少标准差才能被视为离群值。指定 nsigma 为实数标量。该函数通过将本地中位数绝对偏差（MAD）按比例放大 κ= 来估计标准偏差。
数据类型：single

### 输出参量

y - 滤波信号
向量 | 矩阵
过滤后的信号，以与 x 大小相同的向量或矩阵形式返回。
数据类型：single | double

j - 离群点指数
向量 | 矩阵
离群点索引，以与 x 大小相同的向量或矩阵形式返回。
数据类型：逻辑

xmedian - 本地中值
向量 | 矩阵
本地中值，以与 x 大小相同的向量或矩阵形式返回。
数据类型：single

xsigma - 估计标准偏差
向量 | 矩阵
估计标准偏差，以与 x 相同大小的向量或矩阵形式返回。
数据类型：single

## Python函数描述：hampel

函数来源：自定义

### 函数定义：

import numpy as np
def hampel(x, k=7, t0=3):
    """
    Hampel滤波器函数
    
    参数：
    x : numpy.ndarray
        输入信号
    k : int, optional
        窗口半宽度，默认为7
    t0 : float, optional
        阈值因子，默认为3
    返回：
    y : numpy.ndarray
        滤波后的信号
    i : list
        异常值的索引
    xmedian : numpy.ndarray
        每个点的中值
    xsigma : numpy.ndarray
        每个点的中值绝对偏差
    """
    n = len(x)
    y = np.copy(x)
    i = []
    xmedian = np.zeros(n)
    xsigma = np.zeros(n) 
    for j in range(n):
        if j < k or j >= n - k:
            continue
        xk = x[j-k:j+k+1]
        median = np.median(xk)
        sigma = 1.4826 * np.median(np.abs(xk - median))  
        if np.abs(x[j] - median) > t0 * sigma:
            y[j] = median
            i.append(j)
        xmedian[j] = median
        xsigma[j] = sigma
    return y, i, xmedian, xsigma
函数提供了一个完整的Hampel滤波器实现，包括异常值检测、替换以及中值和中值绝对偏差的计算。

### 参数

- x : numpy.ndarray
输入信号，一个一维numpy数组。
- k : int, optional
窗口半宽度，用于计算中值和中值绝对偏差（MAD）。默认值为7。
- t0 : float, optional
阈值因子，用于确定异常值的阈值。默认值为3。

### 返回值

- y : numpy.ndarray
滤波后的信号，一个一维numpy数组，与输入信号x具有相同的长度。
- i : list
异常值的索引列表，包含输入信号x中被替换为中值的异常值的索引。
- xmedian : numpy.ndarray
输入信号x在每个点的中值，一个一维numpy数组，与输入信号x具有相同的长度。
- xsigma : numpy.ndarray
输入信号x在每个点的中值绝对偏差，一个一维numpy数组，与输入信号x具有相同的长度。

### 注意事项

- 

### 函数工作原理

1. 初始化变量
n：获取输入信号x的长度。
y：复制输入信号x以创建一个新的输出信号。
i：创建一个空列表，用于存储异常值的索引。
xmedian 和 xsigma：初始化两个与输入信号x相同长度的零数组，用于存储每个点的中值和中值绝对偏差。
2. 遍历信号
对于信号x中的每个点j：
如果j小于k或大于n-k（k为窗口半宽度），则跳过当前点。
获取当前点j的窗口xk，包括k个前后的点。
计算窗口xk的中值median。
计算窗口xk的中值绝对偏差（MAD）sigma。
3. 异常值检测
如果当前点j的绝对偏差超过阈值t0 * sigma：
在输出信号y中将当前点j替换为窗口xk的中值median。
将当前点j的索引j添加到异常值索引列表i中。
4. 存储中值和中值绝对偏差
在xmedian数组的当前位置存储median。
在xsigma数组的当前位置存储sigma。
5. 返回结果
返回滤波后的信号y、异常值索引列表i、每个点的中值xmedian和每个点的中值绝对偏差xsigma。

### 使用场景

- Hampel滤波器是一种特殊的中值滤波器，设计用于识别和替换信号中的异常值

### 改进建议

- 生成不存在的函数时可以使用多个LLM，并来回将另一个模型的输出结果输入到该模型，最后进行人为修改

## Prompt 1 ： 生成 Python hampel 函数

这是MATLAB示例
```
x = sin(2*pi*(0:99)/100);
x(6) = 2;
x(20) = -2;
[y,i,xmedian,xsigma] = hampel(x);
n = 1:length(x);
plot(n,x)
hold on
plot(n,xmedian-3*xsigma,n,xmedian+3*xsigma)
plot(find(i),x(i),'sk')
hold off
legend('Original signal','Lower limit','Upper limit','Outliers')
```
生成python程序。

### LLM 生成 ： 
以下是将给定的MATLAB代码转换成Python代码的版本：

```
import numpy as np
import matplotlib.pyplot as plt

def hampel(x, k=7, t0=3):
    n = len(x)
    y = np.copy(x)
    i = []
    xmedian = np.zeros(n)
    xsigma = np.zeros(n)
    
    for j in range(n):
        if j < k or j >= n - k:
            continue
        
        xk = x[j-k:j+k+1]
        median = np.median(xk)
        sigma = 1.4826 * np.median(np.abs(xk - median))
        
        if np.abs(x[j] - median) > t0 * sigma:
            y[j] = median
            i.append(j)
        
        xmedian[j] = median
        xsigma[j] = sigma
    
    return y, i, xmedian, xsigma

x = np.sin(2 * np.pi * np.arange(0, 100) / 100)
x[5] = 2
x[19] = -2

y, i, xmedian, xsigma = hampel(x)

n = np.arange(1, len(x) + 1)
plt.plot(n, x, label='Original signal')
plt.plot(n, xmedian - 3 * xsigma, label='Lower limit')
plt.plot(n, xmedian + 3 * xsigma, label='Upper limit')
plt.plot(np.array(i) + 1, x[np.array(i)], 'sk', label='Outliers')
plt.ylim(-2, 2)  # 设置纵轴范围
plt.legend()
plt.show()

```
## Prompt 2 ： 打包 Python hampel 函数
 
 将该hampel函数打包

### LLM生成：Python hampel 函数

import numpy as np
def hampel(x, k=7, t0=3):
    """
    Hampel滤波器函数
    
    参数：
    x : numpy.ndarray
        输入信号
    k : int, optional
        窗口半宽度，默认为7
    t0 : float, optional
        阈值因子，默认为3

    返回：
    y : numpy.ndarray
        滤波后的信号
    i : list
        异常值的索引
    xmedian : numpy.ndarray
        每个点的中值
    xsigma : numpy.ndarray
        每个点的中值绝对偏差
    """
    n = len(x)
    y = np.copy(x)
    i = []
    xmedian = np.zeros(n)
    xsigma = np.zeros(n)
    
    for j in range(n):
        if j < k or j >= n - k:
            continue
        
        xk = x[j-k:j+k+1]
        median = np.median(xk)
        sigma = 1.4826 * np.median(np.abs(xk - median))
        
        if np.abs(x[j] - median) > t0 * sigma:
            y[j] = median
            i.append(j)
        
        xmedian[j] = median
        xsigma[j] = sigma
    
    return y, i, xmedian, xsigma

## Prompt 3 ： 应用生成的 Python hampel 函数做MATLAB示例对应

这是hampel函数的python程序：
import numpy as np
def hampel(x, k=7, t0=3):
    """
    Hampel滤波器函数
    
    参数：
    x : numpy.ndarray
        输入信号
    k : int, optional
        窗口半宽度，默认为7
    t0 : float, optional
        阈值因子，默认为3

    返回：
    y : numpy.ndarray
        滤波后的信号
    i : list
        异常值的索引
    xmedian : numpy.ndarray
        每个点的中值
    xsigma : numpy.ndarray
        每个点的中值绝对偏差
    """
    n = len(x)
    y = np.copy(x)
    i = []
    xmedian = np.zeros(n)
    xsigma = np.zeros(n)
    
    for j in range(n):
        if j < k or j >= n - k:
            continue
        
        xk = x[j-k:j+k+1]
        median = np.median(xk)
        sigma = 1.4826 * np.median(np.abs(xk - median))
        
        if np.abs(x[j] - median) > t0 * sigma:
            y[j] = median
            i.append(j)
        
        xmedian[j] = median
        xsigma[j] = sigma
    
    return y, i, xmedian, xsigma

参考下面的MATLAB代码，给出python程序
x = sin(2*pi*(0:99)/100);
x(6) = 2;
x(20) = -2;
[y,i,xmedian,xsigma] = hampel(x);
n = 1:length(x);
plot(n,x)
hold on
plot(n,xmedian-3*xsigma,n,xmedian+3*xsigma)
plot(find(i),x(i),'sk')
hold off
legend('Original signal','Lower limit','Upper limit','Outliers')

### LLM生成：

你可以按照提供的MATLAB代码，使用numpy和matplotlib来编写相应的Python程序。以下是按照你的要求编写的Python程序：
import numpy as np
import matplotlib.pyplot as plt
def hampel(x, k=7, t0=3):
    """
    Hampel滤波器函数
    
    参数：
    x : numpy.ndarray
        输入信号
    k : int, optional
        窗口半宽度，默认为7
    t0 : float, optional
        阈值因子，默认为3

    返回：
    y : numpy.ndarray
        滤波后的信号
    i : list
        异常值的索引
    xmedian : numpy.ndarray
        每个点的中值
    xsigma : numpy.ndarray
        每个点的中值绝对偏差
    """
    n = len(x)
    y = np.copy(x)
    i = []
    xmedian = np.zeros(n)
    xsigma = np.zeros(n)
    
    for j in range(n):
        if j < k or j >= n - k:
            continue
        
        xk = x[j-k:j+k+1]
        median = np.median(xk)
        sigma = 1.4826 * np.median(np.abs(xk - median))
        
        if np.abs(x[j] - median) > t0 * sigma:
            y[j] = median
            i.append(j)
        
        xmedian[j] = median
        xsigma[j] = sigma
    
    return y, i, xmedian, xsigma
# 生成示例信号
n = 100
x = np.sin(2 * np.pi * np.arange(n) / n)
x[5] = 2
x[19] = -2
# 应用Hampel滤波器
y, i, xmedian, xsigma = hampel(x)
# 绘图
plt.figure(figsize=(10, 6))
plt.plot(np.arange(n), x, label='Original signal')
plt.plot(np.arange(n), xmedian - 3 * xsigma, label='Lower limit')
plt.plot(np.arange(n), xmedian + 3 * xsigma, label='Upper limit')
plt.plot(np.array(i), x[i], 'sk', label='Outliers')
plt.legend()
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Hampel Filter Example')
plt.grid(True)
plt.show()




