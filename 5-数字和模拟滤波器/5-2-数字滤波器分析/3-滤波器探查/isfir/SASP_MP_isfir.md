# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波器分析

## MATLAB函数描述：isfir 

函数来源：[MATLAB isfir](https://ww2.mathworks.cn/help/signal/ref/isfir.html)

### 语法

flag = isfir(d)

### 说明

flag = isfir(d) returns true if a digital filter, d, has a finite impulse response.

### 输入参数

d — Digital filter
digitalFilter object

Digital filter, specified as a digitalFilter object. Use designfilt to generate a digital filter based on frequency-response specifications.

Example: d = designfilt('lowpassiir','FilterOrder',3,'HalfPowerFrequency',0.5) specifies a third-order Butterworth filter with normalized 3 dB frequency 0.5π rad/sample.

### 输出参量

flag — Filter class identification
logical scalar

Filter class identification, returned as a logical scalar.


## Python函数描述：isfir 

函数来源：自定义

### isfir 函数定义：

import numpy as np

def is_fir(b, a):  
    return np.allclose(a, [1.0] + [0.0] * (len(a) - 1)) 

### 参数
- `b`: 滤波器的分子系数。在FIR滤波器中，这些系数描述了滤波器的脉冲响应。
- `a`: 滤波器的分母系数。在FIR滤波器中，这些系数通常只有一个非零元素（即第一个元素为1，其余为0），因为FIR滤波器没有递归部分。

### 返回值
- `True`: 输入的滤波器是FIR滤波器。
- `False`: 输入的滤波器不是FIR滤波器。


## Prompt 1 ： 生成 Python isfir 函数

参考下面MATLAB代码的isfir函数
```
fir = designfilt("highpassfir",FilterOrder=6, ...
    StopbandFrequency=0.3,PassbandFrequency=0.6);
isfirFIR = isfir(fir)

iir = designfilt("highpassiir",FilterOrder=6, ...
    StopbandFrequency=0.3,PassbandFrequency=0.6);
isfirIIR = isfir(iir)

```

完成Python语言isfir函数的编写，便于类似应用调用。

### LLM 生成 ： Python isfir 函数

import numpy as np

def is_fir(b, a):  
    return np.allclose(a, [1.0] + [0.0] * (len(a) - 1))   


## Prompt 2 ： 应用生成的 Python isfir.py 函数做MATLAB示例对应


参考下面Matlab代码，给出Python实现代码：


**Matlab代码**
```
fir = designfilt("highpassfir",FilterOrder=6, ...
    StopbandFrequency=0.3,PassbandFrequency=0.6);
isfirFIR = isfir(fir)

iir = designfilt("highpassiir",FilterOrder=6, ...
    StopbandFrequency=0.3,PassbandFrequency=0.6);
isfirIIR = isfir(iir)
```

这是Python isfir.py
```
import numpy as np

def is_fir(b, a):  
    return np.allclose(a, [1.0] + [0.0] * (len(a) - 1)) 

```

### LLM 生成 ： 

下面这是调用程序
```
import numpy as np  
from scipy import signal  
from isfir import is_fir
  
# 滤波器阶数  
N = 6  
  
# 采样频率（假设为1 Hz）  
Fs = 1.0  
  
# 通带频率和阻带频率（归一化频率）  
Wp = 0.6 / (Fs / 2)  # 通带截止频率  
Ws = 0.3 / (Fs / 2)  # 阻带截止频率  
  
# 设计一个低通FIR滤波器  
b_lowpass = signal.firwin(N+1, Ws, pass_zero=False)  
  
# 将低通滤波器转换为高通滤波器  
b_highpass = np.ones(len(b_lowpass))  
b_highpass[1:] = -b_lowpass[1:]  
  
isfirFIR = is_fir(b_highpass, [1.0])  
print(f"Is FIR filter FIR? {isfirFIR}")  
  

# 设计一个高通IIR滤波器  
N = 6  # 滤波器阶数  
Wn = 0.6  # 截止频率（归一化）  
Fs = 1.0  # 采样频率（归一化）  
  
# 使用butterworth滤波器设计高通滤波器  
b_hp, a_hp = signal.butter(N, Wn, btype='highpass', analog=False) 

# 检查滤波器是否为FIR类型，并输出结果  
isfirIIR = is_fir(b_hp, a_hp)  
print(f"Is IIR filter FIR? {isfirIIR}")  # 直接打印结果
```




