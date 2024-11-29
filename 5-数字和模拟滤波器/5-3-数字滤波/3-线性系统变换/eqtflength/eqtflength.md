# 信号处理仿真与应用 - 数字和模拟滤波器 - 线性系统变换

## MATLAB函数描述：eqtflength 

- 函数来源：[MATLAB envelope](https://ww2.mathworks.cn/help/signal/ref/eqtflength.html)

### 语法

- [b,a] = eqtflength(num,den)
- [b,a,n,m] = eqtflength(num,den)

### 说明
- [b,a] = eqtflength(num,den)，使输出b，a的向量长度相等。
- [b,a,n,m] = eqtflength(num,den)，使输出b，a的向量长度相等，n等于向量num的阶数，m等于den的阶数。
、
### 输入参数

- num — 输入序列
向量 
示例: num=[1 2 0 3 ]
数据类型: double
支持复数；对

- den — 输入序列
向量 
示例: den=[0 2 0 0 ]
数据类型: double
支持复数；对



### 输出参量

- b 
行向量
解释：b=num（有必要时对num向量后面增添0或者减少0使b和a向量一样长）。

- a 
行向量
解释：a=den（有必要时对den向量后面增添0或者减少0使a和b向量一样长）。

- n
num的阶数
例子：num=[1 0  2  0]，0s^3+2s^2+0s^1+1s^0，则n=2。

- m
den的阶数
例子：den=[1 2  3  4]，4s^3+3s^2+2s^1+1s^0，则n=3。


## Python函数描述：eqtflength

- 函数来源：自定义

### 等效长度函数定义：

import numpy as np

def eqtflength(num, den):
    # Determine lengths of numerator and denominator polynomials
    num_length = len(num)
    den_length = len(den)

    # Adjust lengths of numerator and denominator polynomials to be equal
    max_length = max(num_length, den_length)
    num = np.pad(num, (0, max_length - num_length))
    den = np.pad(den, (0, max_length - den_length))

    # Compute orders of numerator and denominator polynomials
    n = num_length - 1
    m = den_length - 1

    return num, den, n, m



这段代码定义了一个名为 `eqtflength` 的函数,这段代码演示了如何调整分子和分母多项式的长度，以使它们相等，并计算它们的阶数。。

### 参数
- `num`: 输入信号，一维向量。
- `den`: 输入信号，一维向量。

### 返回值

- `num`: 使输入num和输入den长度相等。
- `den`: 使输入num和输入den长度相等。
- `n`: 输入num阶数。
- `m`: 输入den阶数。

### 函数工作原理
1. 确定分子和分母多项式的长度： 首先，代码使用 len() 函数确定分子和分母多项式的长度，即它们包含的项数。
2. 调整多项式的长度： 然后，代码比较分子和分母多项式的长度，找到它们中的最大长度。接着，使用 NumPy 的 pad() 函数来调整分子和分母多项式的长度，使它们都与最大长度相等。如果一个多项式比另一个多项式短，则在其末尾填充零项，直到长度相等。
3. 计算多项式的阶数： 最后，代码根据调整后的分子和分母多项式的长度，计算它们的阶数。分子多项式的阶数等于其长度减去1，因为阶数是最高次幂的指数。同样地，分母多项式的阶数也等于其长度减去1。

### 使用场景
1. 分析和设计数字控制系统： 在数字控制系统的分析和设计过程中，经常需要对离散时间传递函数进行操作。"eqtflength" 函数可以确保分子和分母多项式的长度相同，这在一些分析和设计方法中是必要的。
2. 数字滤波器设计： 在数字滤波器设计中，通常需要处理离散时间传递函数。通过使用 "eqtflength" 函数，可以确保分子和分母多项式的长度相同，从而简化滤波器设计的过程。
3.系统模型简化： 在建立系统模型时，可能会遇到分子和分母多项式长度不匹配的情况。"eqtflength" 函数可以用于调整多项式的长度，使它们相等，从而简化系统模型的表示和分析。
4.频域分析： 在频域分析中，需要对系统的传递函数进行操作。"eqtflength" 函数可以确保传递函数的分子和分母多项式长度相同，以便进行频域分析和频率响应计算。
### 改进建议


## Prompt 1 ： 生成 Python eqtflength(num,den) 函数

参考下面MATLAB代码的envelope函数
```
num = [0 0 2];
den = [4 0 3 -1];

[b,a,n,m] = eqtflength(num,den)
```

和我们采用Python语言实现该函数，
```
num = np.array([0, 0, 2])   # Numerator polynomial: 2s^2
den = np.array([4, 0, 3, -1])  # Denominator polynomial: 4s^3 + 3s^2 - s
b, a, n, m = eqtflength(num, den)
print("Modified numerator polynomial:", b)
print("Modified denominator polynomial:", a)
print("Numerator order:", n)
print("Denominator order:", m)
```



```python

```
