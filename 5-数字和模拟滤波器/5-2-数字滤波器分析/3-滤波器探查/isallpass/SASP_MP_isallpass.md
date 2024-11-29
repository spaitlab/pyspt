# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波器分析

## MATLAB函数描述：isallpass 

函数来源：[MATLAB isallpass](https://ww2.mathworks.cn/help/signal/ref/isallpass.html)

### 语法

flag = isallpass(b,a)
flag = isallpass(sos)
flag = isallpass(d)
flag = isallpass(...,tol)

### 说明

flag = isallpass(b,a) returns a logical output, flag, equal to true if the filter specified by numerator coefficients, b, and denominator coefficients, a, is an allpass filter. If the filter is not an allpass filter, flag is equal to false.

flag = isallpass(sos) returns true if the filter specified by second order sections matrix, sos, is an allpass filter. sos is a K-by-6 matrix, where the number of sections, K, must be greater than or equal to 2. Each row of sos corresponds to the coefficients of a second order (biquad) filter. The ith row of the sos matrix corresponds to [bi(1) bi(2) bi(3) ai(1) ai(2) ai(3)].

flag = isallpass(d) returns true if the digital filter, d, is an allpass filter. Use designfilt to generate d based on frequency-response specifications.

flag = isallpass(...,tol) uses the tolerance, tol, to determine when two numbers are close enough to be considered equal. If not specified, tol, defaults to eps^(2/3). Specifying a tolerance may be most helpful in fixed-point allpass filters.



## Python函数描述：isallpass 

函数来源：自定义

### isallpass函数定义：

def is_allpass_filter(b, a):
    # 计算滤波器的零点和极点
    zeros = np.roots(b)
    poles = np.roots(a)
    
    # 判断是否为全通滤波器
    if len(zeros) != len(poles):
        return False
    
    reciprocal_zeros = 1 / np.conj(zeros)
    reciprocal_zeros = np.sort(reciprocal_zeros)
    poles = np.sort(poles)
    
    return np.allclose(reciprocal_zeros, poles, rtol=1e-5, atol=1e-8)

### 参数
- `b`: 滤波器的分子系数。
- `a`: 滤波器的分母系数。

### 返回值
- `True`: 输入的滤波器是全通滤波器。
- `False`:输入的滤波器不是全通滤波器。

## Prompt 1 ： 生成 Python isallpass 函数

参考下面MATLAB代码的isallpass函数
```
b = [1/3 1/4 1/5 1];
a = fliplr(b); 
flag = isallpass(b,a)
```
完成Python语言isallpass函数的编写，便于类似应用调用。

### LLM 生成 ： Python isallpass 函数

import numpy as np

def is_allpass_filter(b, a):
    # 计算滤波器的零点和极点
    zeros = np.roots(b)
    poles = np.roots(a)
    
    # 判断是否为全通滤波器
    if len(zeros) != len(poles):
        return False
    
    reciprocal_zeros = 1 / np.conj(zeros)
    reciprocal_zeros = np.sort(reciprocal_zeros)
    poles = np.sort(poles)
    
    return np.allclose(reciprocal_zeros, poles, rtol=1e-5, atol=1e-8)


## Prompt 2 ： 应用生成的 Python is_allpass_filter.py 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：


**Matlab代码**
```
b = [1/3 1/4 1/5 1];
a = fliplr(b); 
flag = isallpass(b,a)

k = [1/2 1/3 1/4 1/5];
[b,a] = latc2tf(k,"allpass");
flag_isallpass = isallpass(b,a)
```


### LLM 生成 ： 

下面这是调用程序
```
import numpy as np
from is_allpass_filter import is_allpass_filter

# 示例
b = np.array([1/3, 1/4, 1/5, 1])
a = np.flip(b)

is_allpass = is_allpass_filter(b, a)

if is_allpass:
    print("The filter is an allpass filter.")
else:
    print("The filter is not an allpass filter.")
```

```
import numpy as np  
from is_allpass_filter import is_allpass_filter
   
b = np.array([0.2, 0.4, 0.6, 0.8, 1])  
a = np.array([1, 0.8, 0.6, 0.4, 0.2]) 
is_allpass = is_allpass_filter(b, a)

if is_allpass:
    print("The filter is an allpass filter.")
else:
    print("The filter is not an allpass filter.")
```

