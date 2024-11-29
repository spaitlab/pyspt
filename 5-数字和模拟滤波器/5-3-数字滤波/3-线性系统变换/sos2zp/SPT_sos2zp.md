# 信号处理仿真和应用-数字和模拟滤波器-线性系统变换-sos2zp
## MATLAB函数描述：sos2zp
函数来源：[[MATLAB sos2zp](https://www.mathworks.com/help/signal/ref/sos2zp.html?s_tid=doc_ta)]
### 语法
[z,p,k]=sos2zp(sos)  
[z,p,k]=sos2zp(sos,g)
### 使用说明
[z,p,k]=sos2zp(sos)会返回数字滤波器的零点、极点、增益  
[z,p,k]=sos2zp(sos,g)会返回具有系统增益g的数字滤波器的零点、极点、增益  
### 输入参数
+ sos-二阶节数字滤波器的系数，用L*6的矩阵表示，L代表二阶节数字滤波器的个数，比如两个二阶节数字滤波器参数形式如下：
$$
sos = \left[ \begin{matrix} b_{01} & b_{11}  & b_{21} & a_{01} & a_{11} & a_{21}\\ b_{02} & b_{12}  & b_{22} & a_{02} & a_{12} & a_{22} \end{matrix} \right]
$$
其中b<sub>01</sub>,b<sub>11</sub>,b<sub>21</sub>为第一个二阶节数字滤波器分子系数，a<sub>01</sub>,a<sub>11</sub>,a<sub>21</sub>为第一个二阶节数字滤波器分母系数。同理第二行为第二个二阶节数字滤波器的系数。

+ g-数字滤波器系统增益，一个实标量。

### 输出参数
+ z-零点，返回系统零点的矩阵；
+ p-极点，返回系统极点的矩阵；
+ k-增益，返回系统增益的一个实标量。

## Python函数描述：sos2zp
函数来源：自定义  
### Python中自定义的sos2zp函数
```python
from scipy.signal import sos2zpk as original_sos2zpk

def sos2zpk(sos, gain=1):

    Z, P, K = original_sos2zpk(sos)
    K *= gain
    return Z, P, K
```

### 输入参数
+ sos：数字滤波器的二阶节参数，以L*6大小的数组输入；
+ g:系统增益，默认为1，增益为1时可以不输入。

### 输出参数
+ z-零点，返回系统零点的向量；
+ p-极点，返回系统极点的向量；
+ k-增益，返回系统增益的一个实标量。

### Python函数工作原理
1. 借助原有函数sos2zpk将二阶节参数转换为零点、极点、增益形式；
2. 再增益乘以g得到零极点形式下数字滤波器总增益k。

### 使用场景
将数字滤波器二阶节参数转化为零点、极点、增益的形式。

### 改进建议
1. Python和MATLAB中保留小数位数有所不同，后续可以规定保留有效数字让显示更简洁规范。

## Prompt 1 生成Python中sos2zp函数
在MATLAB中sos2zp函数代码如下：
```
% 给定二阶节参数
sos = [1 1 1 1 0 -1; 
      -2 3 1 1 10 1];

% 使用sos2zp函数转换为零极点表示形式
[Z, P, K] = sos2zp(sos，2);

% 输出结果
disp('零点：');
disp(Z);
disp('极点：');
disp(P);
disp('增益：');
disp(K);
```
在Python中实现MATLAB中sos2zp过程
```Python
import numpy as np
from scipy.signal import sos2zpk

# 给定二阶节参数
sos = np.array([
                [1, 1, 1, 1, 0, -1], 
                [-2, 3, 1, 1, 10, 1]
                ])

# 使用sos2zpk函数转换为零极点表示形式
Z, P, K = sos2zpk(sos)

# 指定系统增益
g = 2
# 得到零极点形式下增益
K *= g

print('零点：', Z)
print('极点：', P)
print('增益：', K)
```
### 定义Python中sos2zp函数
```Python
from scipy.signal import sos2zpk

def sos2zp(sos, gain=1):
    # 使用原始的sos2zpk函数转换为零极点表示形式
    Z, P, K =sos2zpk(sos)
    # 得到零极点形势下增益
    K *= gain
    
    return Z, P, K
```
## Prompt 2 使用自定义的Python中sos2zp函数与MATLAB示例对应
给出待处理数据：
```Python
# 给定二阶节参数
sos = np.array([
                [1, 1, 1, 1, 0, -1], 
                [-2, 3, 1, 1, 10, 1]
                ])

# 给出二阶节系数下增益
g = 2
```
Python自定义的sos2zp函数：
```Python
from scipy.signal import sos2zpk

def sos2zp(sos, gain=1):
    # 使用原始的sos2zpk函数转换为零极点表示形式
    Z, P, K = sos2zpk(sos)
    # 得到零极点形势下增益
    K *= gain
    
    return Z, P, K
```
### 使用Python完成示例
```Python
import numpy as np
from scipy.signal import sos2zpk

def sos2zp(sos, gain=1):
    # 使用原始的sos2zpk函数转换为零极点表示形式
    Z, P, K = sos2zpk(sos)
    # 得到零极点形式下增益
    K *= gain
    
    return Z, P, K

# 给定二阶节参数
sos = np.array([
                [1, 1, 1, 1, 0, -1], 
                [-2, 3, 1, 1, 10, 1]
                ])

# 给出二阶节系数下增益
g = 2

# 使用自定义函数转换为零极点、增益表示形式
Z, P, K = sos2zp(sos, g)

print('零点：', Z)
print('极点：', P)
print('增益：', K)
```

## Prompt 3 使用自定义的Python中sos2zp.py与MATLAB示例对应
MATLAB中处理数据如下：
```
% 给定二阶节参数
sos = [1 1 1 1 0 -1; 
      -2 3 1 1 10 1];

% 使用sos2zp函数转换为零极点表示形式
[Z, P, K] = sos2zp(sos,2);

disp('零点：');
disp(Z);
disp('极点：');
disp(P);
disp('增益：');
disp(K);
```
Python中sos2zp.py文件如下：
```Python
from scipy.signal import sos2zpk

def sos2zp(sos, gain=1):
    # 使用原始的sos2zpk函数转换为零极点表示形式
    Z, P, K = sos2zpk(sos)
    # 得到零极点形式下增益
    K *= gain
    
    return Z, P, K
```
Python中通过调用sos2zp.py实现程序
```Python
import numpy as np
from sos2zp import sos2zp

# 给定二阶节参数
sos = np.array([
                [1, 1, 1, 1, 0, -1], 
                [-2, 3, 1, 1, 10, 1]
                ])

# 给出二阶节系数下增益
g = 2

# 使用自定义函数转换为零极点、增益表示形式
Z, P, K = sos2zp(sos, g)

print('零点：', Z)
print('极点：', P)
print('增益：', K)
```