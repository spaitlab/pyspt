# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波器分析

## MATLAB函数描述：islinphase 

函数来源：[MATLAB islinphase](https://ww2.mathworks.cn/help/signal/ref/islinphase.html)

### 语法

flag = islinphase(b,a)
flag = islinphase(sos)
flag = islinphase(d)
flag = islinphase(...,tol)

### 说明

flag = islinphase(b,a) returns a logical output, flag, equal to true if the filter coefficients in b and a define a linear phase filter. flag is equal to false if the filter does not have linear phase.

flag = islinphase(sos) returns true if the filter specified by second order sections matrix, sos, has linear phase. sos is a K-by-6 matrix, where the number of sections, K, must be greater than or equal to 2. Each row of sos corresponds to the coefficients of a second order (biquad) filter. The ith row of the sos matrix corresponds to [bi(1) bi(2) bi(3) ai(1) ai(2) ai(3)].

flag = islinphase(d) returns true if the digital filter, d, has linear phase. Use designfilt to generate d based on frequency-response specifications.

flag = islinphase(...,tol) uses the tolerance, tol, to determine when two numbers are close enough to be considered equal. If not specified, tol, defaults to eps^(2/3).


## Python函数描述：islinphase


线性相位的定义：滤波器对不同频率的正弦波所产生的相移和正弦波的频率具有线性关系。

线性相位的因果FIR系统的单位序列响应特性：单位序列响应满足h(n)= ± h(N − 1 − n) ，其中 N 代表序列的长度。

h(n) = − h(N − 1 − n) 代表h(n)关于（N-1）/2奇对称                     
h(n) =   h(N − 1 − n) 代表h(n)关于（N-1）/2偶对称     

函数来源：自定义

### islinphase函数定义：

def is_linear_phase(b):
    N = len(b)
    
    # 序列长度为偶数
    if N % 2 == 0:
        if np.allclose(b, b[::-1]):
            return True  # 滤波器系数偶对称，属于线性相位滤波器
        elif np.allclose(b, -b[::-1]):
            return True  # 滤波器系数奇对称，属于线性相位滤波器
        return False

    elif N % 2 != 0:
    # 序列长度为奇数
        middle_index = N // 2  
        # 检查是否偶对称或奇对称  
        if np.allclose(b[:middle_index], b[middle_index + 1:][::-1]):  
            return True  
        elif np.allclose(b[:middle_index], -b[middle_index + 1:][::-1]):  
            return True  
        else:  
            return False
        
    else:  
        return False   


### 参数
- `b`: FIR滤波器的系数。FIR滤波器是一种只包含零点而没有极点的滤波器，其输出仅依赖于当前的输入和过去的输入。

### 返回值
- `True`: 输入的滤波器是线性相位滤波器。
- `False`: 输入的滤波器不是线性相位滤波器。


## Prompt 1 ： 生成 Python islinphase 函数

参考下面MATLAB代码的islinphase函数
```
d = designfilt('lowpassfir','DesignMethod','window', ...
    'FilterOrder',10,'CutoffFrequency',0.55);
flag = islinphase(d)
```

完成Python语言islinphase函数的编写，便于类似应用调用。

### LLM 生成 ： Python islinphase 函数

def is_linear_phase(b):
    N = len(b)
    
    # 序列长度为偶数
    if N % 2 == 0:
        if np.allclose(b, b[::-1]):
            return True  # 滤波器系数偶对称，属于线性相位滤波器
        elif np.allclose(b, -b[::-1]):
            return True  # 滤波器系数奇对称，属于线性相位滤波器
        return False

    elif N % 2 != 0:
    # 序列长度为奇数
        middle_index = N // 2  
        # 检查是否偶对称或奇对称  
        if np.allclose(b[:middle_index], b[middle_index + 1:][::-1]):  
            return True  
        elif np.allclose(b[:middle_index], -b[middle_index + 1:][::-1]):  
            return True  
        else:  
            return False
        
    else:  
        return False   


## Prompt 2 ： 应用生成的 Python islinphase 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：

**Matlab代码**
```
d = designfilt('lowpassfir','DesignMethod','window', ...
    'FilterOrder',10,'CutoffFrequency',0.55);
flag = islinphase(d)

[phs,w] = phasez(d);

plot(w/pi,phs)
xlabel('Frequency \omega/\pi')
ylabel('Phase')
```


### LLM 生成 ： 

import numpy as np  
import matplotlib.pyplot as plt  
from scipy import signal  

def is_linear_phase(b):
    N = len(b)
    
    # 序列长度为偶数
    if N % 2 == 0:
        if np.allclose(b, b[::-1]):
            return True  # 滤波器系数偶对称，属于线性相位滤波器
        elif np.allclose(b, -b[::-1]):
            return True  # 滤波器系数奇对称，属于线性相位滤波器
        return False
        
    elif N % 2 != 0:
    # 序列长度为奇数
        middle_index = N // 2  
        # 检查是否偶对称或奇对称  
        if np.allclose(b[:middle_index], b[middle_index + 1:][::-1]):  
            return True  
        elif np.allclose(b[:middle_index], -b[middle_index + 1:][::-1]):  
            return True  
        else:  
            return False
        
    else:  
        return False   
  
# 设计参数  
N = 11  # 滤波器阶数加1（因为firwin的N是滤波器系数的数量）  
cutoff_freq = 0.55  # 归一化截止频率 (0到1之间)  
window = 'hamming'  # 窗函数类型  
  
# 使用firwin设计FIR滤波器  
b = signal.firwin(N, cutoff_freq, window=window)  
print(b)  
# 滤波器系数a总是为[1]对于FIR滤波器  
a = [1]  
  
# 计算并绘制频率响应  
w, h = signal.freqz(b, a)  
frequencies = w / np.pi  # 将角频率转换为归一化频率  
# amplitudes = 20 * np.log10(np.abs(h))  # 转换为分贝  
  
# 绘制幅度响应  
# plt.figure()  
# plt.plot(frequencies, amplitudes)  
# plt.title('FIR Lowpass Filter Frequency Response')  
# plt.xlabel('Normalized Frequency (xπrad/sample)')  
# plt.ylabel('Amplitude (dB)')  
# plt.grid(True)  
# plt.ylim([-100, 5])  
# plt.axvline(x=cutoff_freq, color='r', linestyle='--', label='Cutoff Frequency')  
# plt.legend()  
  
# 绘制相位响应  
phase = np.unwrap(np.angle(h))  
plt.figure()  
plt.plot(frequencies, phase)  
plt.title('FIR Lowpass Filter Phase Response')  
plt.xlabel('Normalized Frequency (xπrad/sample)')  
plt.ylabel('Phase (radians)')  
plt.grid(True)  
plt.axvline(x=cutoff_freq, color='r', linestyle='--', label='Cutoff Frequency')  
plt.legend()  


# 函数示例使用  
print(is_linear_phase(b))  # 应该输出True 

