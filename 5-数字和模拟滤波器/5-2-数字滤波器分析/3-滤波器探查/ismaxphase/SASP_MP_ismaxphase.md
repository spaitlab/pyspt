# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波器分析

## MATLAB函数描述：ismaxphase 

函数来源：[MATLAB ismaxphase](https://ww2.mathworks.cn/help/signal/ref/ismaxphase.html)

### 语法

flag = ismaxphase(b,a)
flag = ismaxphase(sos)
flag = ismaxphase(d)
flag = ismaxphase(...,tol)

### 说明

flag = ismaxphase(b,a) returns a logical output, flag, equal to true if the filter specified by numerator coefficients, b, and denominator coefficients, a, is a maximum phase filter.

flag = ismaxphase(sos) returns true if the filter specified by second order sections matrix, sos, is a maximum phase filter. sos is a K-by-6 matrix, where the number of sections, K, must be greater than or equal to 2. Each row of sos corresponds to the coefficients of a second order (biquad) filter. The ith row of the sos matrix corresponds to [bi(1) bi(2) bi(3) ai(1) ai(2) ai(3)].

flag = ismaxphase(d) returns true if the digital filter, d, has maximum phase. Use designfilt to generate d based on frequency-response specifications.

flag = ismaxphase(...,tol) uses the tolerance, tol, to determine when two numbers are close enough to be considered equal. If not specified, tol, defaults to eps^(2/3).


## Python函数描述：ismaxphase 

函数来源：自定义
存在问题：可能是由于Python代码精度的限制，所以存在判断误差。

### ismaxphase函数定义：

def ismaxphase(b, a, tol=np.finfo(float).eps**(2/3)):
    # 判断是否为最大相位滤波器
    if isinstance(b, np.ndarray) and isinstance(a, np.ndarray):
        roots_a = np.roots(a)
        roots_b = np.roots(b)
        if np.all(np.real(roots_a) < 0) and np.all(np.abs(roots_b) < 1 + tol):
            return True
        else:
            return False
    elif isinstance(b, np.ndarray) and a is None:
        sos = b
        if sos.shape[1] != 6 or sos.shape[0] < 2:
            raise ValueError("Invalid sos matrix shape")
        
        for i in range(sos.shape[0]):
            bi = sos[i, :3]
            ai = sos[i, 3:]
            if not ismaxphase(bi, ai, tol):
                return False
        return True
    elif b is not None and a is None:
        d = b
        b, a = d.num, d.den
        return ismaxphase(b, a, tol)
    else:
        raise ValueError("Invalid input")



### 参数
- `b`: 滤波器的分子系数。对于IIR滤波器，b 是一个一维数组；对于SOS（二阶段）形式的滤波器，b 是一个二维数组，其中每一行表示一个SOS节的分子系数。

- `a`: 滤波器的分母系数。对于IIR滤波器，a 是一个一维数组；对于SOS形式的滤波器，a 也是一个一维数组，其中每一行表示一个SOS节的分母系数。如果a是None，并且b是一个SOS矩阵，则函数将处理SOS矩阵。

- `tol`: 可选参数，表示在判断滤波器系数是否在单位圆内时所使用的容差值。默认值为np.finfo(float).eps**(2/3)，它是一个非常小的数，用于处理浮点数的精度问题。

### 返回值
- `True`: 输入的滤波器是最大相位滤波器。
- `False`:输入的滤波器不是最大相位滤波器。


## Prompt 1 ： 生成 Python ismaxphase 函数

参考下面MATLAB代码的ismaxphase函数
```
b = single([1 -0.9999]);
a = single([1 0.45]);
max_flag1 = ismaxphase(b,a)
max_flag2 = ismaxphase(b,a,1e-3)
```

完成Python语言ismaxphase函数的编写，便于类似应用调用。

### LLM 生成 ： Python ismaxphase 函数

import numpy as np

def ismaxphase(b, a, tol=np.finfo(float).eps**(2/3)):
    # 判断是否为最大相位滤波器
    if isinstance(b, np.ndarray) and isinstance(a, np.ndarray):
        roots_a = np.roots(a)
        roots_b = np.roots(b)
        if np.all(np.real(roots_a) < 0) and np.all(np.abs(roots_b) < 1 + tol):
            return True
        else:
            return False
    elif isinstance(b, np.ndarray) and a is None:
        sos = b
        if sos.shape[1] != 6 or sos.shape[0] < 2:
            raise ValueError("Invalid sos matrix shape")
        
        for i in range(sos.shape[0]):
            bi = sos[i, :3]
            ai = sos[i, 3:]
            if not ismaxphase(bi, ai, tol):
                return False
        return True
    elif b is not None and a is None:
        d = b
        b, a = d.num, d.den
        return ismaxphase(b, a, tol)
    else:
        raise ValueError("Invalid input")



## Prompt 2 ： 应用生成的 Python is_max_phase.py 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：

**Matlab代码**
```
b = single([1 -0.9999]);
a = single([1 0.45]);
max_flag1 = ismaxphase(b,a)
max_flag2 = ismaxphase(b,a,1e-3)
```


### LLM 生成 ： 

下面这是调用程序
```
import numpy as np

from is_max_phase import ismaxphase


# 示例案例
b = np.array([1, -0.9999], dtype=np.float32)
a = np.array([1, 0.45], dtype=np.float32)
max_flag1 = ismaxphase(b, a)
print(max_flag1)

max_flag2 = ismaxphase(b, a, 1e-1000)
print(max_flag2)
```


直接调用函数
```
import numpy as np  
from scipy.signal import freqz

def ismaxphase(b, a=None, tol=np.finfo(float).eps**(2/3)):  
    """  
    判断滤波器是否为最大相位滤波器。  
      
    参数:  
    b : np.ndarray  
        滤波器的分子系数。  
    a : np.ndarray, 可选  
        滤波器的分母系数。如果未提供，则假设为FIR滤波器（a=[1]）。  
    tol : float, 可选  
        容差值，用于确定相位是否足够接近零来被认为是最大相位。  
      
    返回:  
    f : bool  
        如果滤波器是最大相位滤波器，返回True；否则返回False。  
    """  
    # 如果未提供分母系数，则假设为FIR滤波器  
    if a is None:  
        a = np.array([1])  
  
    # 检查输入是否为SOS矩阵  
    if b.ndim == 2 and b.shape[1] == 6:  
        # 输入是SOS矩阵，检查每个SOS部分是否为最大相位  
        for bi, ai in b:  
            w, h = freqz(bi, ai)  
            phase = np.unwrap(np.angle(h))  
            if not np.all(phase >= -tol):  
                return False  
        return True  
    else:  
        # 输入是分子和分母系数  
        w, h = freqz(b, a)  
        phase = np.unwrap(np.angle(h))  
        # 检查相位是否非负  
        return np.all(phase >= -tol)   

# 定义滤波器系数  
b = np.array([1, -0.9999], dtype=np.float32)  
a = np.array([1, 0.45], dtype=np.float32)  
  
# 使用默认容差值判断是否为最大相位滤波器  
max_flag1 = ismaxphase(b, a)  
print("Is the filter maximum phase with default tolerance?", max_flag1)  # 应该是 False  
  
# 使用指定的容差值1e-3判断是否为最大相位滤波器  
max_flag2 = ismaxphase(b, a, 1e-3)  
print("Is the filter maximum phase with tolerance of 1e-3?", max_flag2)  # 应该是 True
```

