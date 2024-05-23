# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波器分析

## MATLAB函数描述：firtype 

函数来源：[MATLAB firtype](https://ww2.mathworks.cn/help/signal/ref/firtype.html)

### 语法

t = firtype(b)
t = firtype(d)

### 说明

t = firtype(b) 确定系数 b 的FIR滤波器的类型 t 。 t 可以为1、2、3或4。滤波器必须是实的，并且具有线性相位。
t = firtype(d) 确定FIR滤波器 d 的类型 t 。 t 可以为1、2、3或4。滤波器必须是实的，并且具有线性相位。

### 输入参数

b — 滤波系数
向量 
FIR滤波器的滤波系数，指定为双单精度或单精度实值行向量或列向量。
数据类型: double | single

d — FIR滤波器
digitalFilter 对象
FIR滤波器，指定为 digitalFilter 对象。使用  designfilt 生成一个基于频率响应规范的数字滤波器。

### 输出参量

t — 滤波器类型
1 | 2 | 3 | 4
滤波器类型，返回为1、2、3或4。滤波器类型定义如下:

类型1 -偶阶对称系数

类型2 -奇阶对称系数

类型3 -偶阶反对称系数

类型4 -奇阶反对称系数



## Python函数描述：firtype

函数来源：自定义

### firetype函数定义：

# 定义firtype函数
def firtype(b):
    N = len(b)
    
    # 序列长度为偶数，则为奇阶(序列长度=阶数+1)
    if N % 2 == 0:
        if np.allclose(b, b[::-1]):
            return 2  # 奇阶对称，type2
        elif np.allclose(b, -b[::-1]):
            return 4  # 奇阶反对称，type4
        return False 
    
    elif N % 2 != 0:
    # 序列长度为奇数，则为偶阶
        middle_index = N // 2   
        if np.allclose(b[:middle_index], b[middle_index + 1:][::-1]):  # 偶阶对称，type1
            return 1 
        elif np.allclose(b[:middle_index], -b[middle_index + 1:][::-1]):  # 偶阶反对称，type3
            return 3  
        else:  
            return False
        
    else:  
        return False   


### 参数
- `b`: 滤波器系数。

### 返回值
- `1`: Even-order symmetric coefficients
- `2`: Odd-order symmetric coefficients
- `3`: Even-order antisymmetric coefficients
- `4`: Odd-order antisymmetric coefficients

## Prompt 1 ： 生成 Python firtype 函数

参考下面MATLAB代码的firtype函数
```
subplot(2,1,1)
b = fir1(8,0.5);
impz(b), title(['Type ' int2str(firtype(b))])

subplot(2,1,2)
b = fir1(9,0.5);
impz(b), title(['Type ' int2str(firtype(b))])
```

```
subplot(2,1,1)
b = firpm(8,[0.2 0.8],[1 1],'hilbert');
impz(b), title(['Type ' int2str(firtype(b))])


subplot(2,1,2)
b = firpm(9,[0.2 0.8],[1 1],'hilbert');
impz(b), title(['Type ' int2str(firtype(b))])
```
完成Python语言firtype函数的编写，便于类似应用调用。

### LLM 生成 ： Python firtype 函数

import numpy as np 

def firtype(b):
    N = len(b)
    
    # 序列长度为偶数，则为奇阶(序列长度=阶数+1)
    if N % 2 == 0:
        if np.allclose(b, b[::-1]):
            return 2  # 奇阶对称，type2
        elif np.allclose(b, -b[::-1]):
            return 4  # 奇阶反对称，type4
        return False 
    
    elif N % 2 != 0:
    # 序列长度为奇数，则为偶阶
        middle_index = N // 2   
        if np.allclose(b[:middle_index], b[middle_index + 1:][::-1]):  # 偶阶对称，type1
            return 1 
        elif np.allclose(b[:middle_index], -b[middle_index + 1:][::-1]):  # 偶阶反对称，type3
            return 3  
        else:  
            return False
        
    else:  
        return False   
     


## Prompt 2 ： 应用生成的 Python firtype.py 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：


**Matlab代码**
```
subplot(2,1,1)
b = fir1(8,0.5);
impz(b), title(['Type ' int2str(firtype(b))])

subplot(2,1,2)
b = fir1(9,0.5);
impz(b), title(['Type ' int2str(firtype(b))])
```
```
subplot(2,1,1)
b = firpm(8,[0.2 0.8],[1 1],'hilbert');
impz(b), title(['Type ' int2str(firtype(b))])


subplot(2,1,2)
b = firpm(9,[0.2 0.8],[1 1],'hilbert');
impz(b), title(['Type ' int2str(firtype(b))])
```

这是Python firtype.py
```
import numpy as np 

def firtype(b):
    N = len(b)
    
    # 序列长度为偶数，则为奇阶(序列长度=阶数+1)
    if N % 2 == 0:
        if np.allclose(b, b[::-1]):
            return 2  # 奇阶对称，type2
        elif np.allclose(b, -b[::-1]):
            return 4  # 奇阶反对称，type4
        return False 
    
    elif N % 2 != 0:
    # 序列长度为奇数，则为偶阶
        middle_index = N // 2   
        if np.allclose(b[:middle_index], b[middle_index + 1:][::-1]):  # 偶阶对称，type1
            return 1 
        elif np.allclose(b[:middle_index], -b[middle_index + 1:][::-1]):  # 偶阶反对称，type3
            return 3  
        else:  
            return False
        
    else:  
        return False   
     
```


### LLM 生成 ： 

下面这是调用程序
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from firtype import firtype
 

# 定义fir1函数
def fir1(numtaps, cutoff):
    return signal.firwin(numtaps+1, cutoff)

# 第一个案例
plt.subplot(2, 1, 1)
b1 = fir1(8, 0.5)
print(b1)
plt.stem(b1)
plt.title('Type ' + str(firtype(b1)))

# 第二个案例
plt.subplot(2, 1, 2)
b2 = fir1(9, 0.5)
print(b2)
plt.stem(b2)
plt.title('Type ' + str(firtype(b2)))

```





