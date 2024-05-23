# 信号处理仿真和应用-数字和模拟滤波器-线性系统变换-ss2sos
## MATLAB函数描述：ss2sos
函数来源：[[MATLAB ss2sos](https://ww2.mathworks.cn/help/signal/ref/ss2sos.html?s_tid=doc_ta)]
### 语法
[sos,g]=ss2sos(A,B,C,D)  
[sos,g]=ss2sos(A,B,C,D,iu)   
[sos,g]=ss2sos(A,B,C,D,ordei)   
[sos,g]=ss2sos(A,B,C,D,iu,order)   
[sos,g]=ss2sos(A,B,C,D,iu,order,scale)   
sos=ss2sos( ) 
### 使用说明
[sos,g]=ss2sos(A,B,C,D)返回数字滤波器状态空间形式的二阶节系数形式，系统必须为单输出系统  
[sos,g]=ss2sos(A,B,C,D,iu)返回数字滤波器状态空间形式的二阶节系数形式，其中iu用于指定输入的未知变量索引，通常用于多输入系统  
[sos,g]=ss2sos(A,B,C,D,ordei)返回数字滤波器状态空间形式的二阶节系数形式，其中order用于指定所需二阶节滤波器的顺序   
[sos,g]=ss2sos(A,B,C,D,iu,order)返回数字滤波器状态空间形式的二阶节系数形式，其中iu用于指定输入的未知变量索引，order用于指定所需二阶节滤波器的顺序  
[sos,g]=ss2sos(A,B,C,D,iu,order,scale)返回数字滤波器状态空间形式的二阶节系数形式，其中scale表示是否对输入信号进行缩放，允许同时指定输入变量索引、二阶节滤波器顺序和是否对信号缩放  
sos=ss2sos( )返回数字滤波器状态空间形式的二阶节系数形式，并在第一个二阶节系数中加入系统增益g
### 输入参数
+ A-状态矩阵，如果系统具有p个输入和q个输出，并由n个状态变量描述，则状态矩阵大小为n*n；
+ B-输入矩阵，如果系统具有p个输入和q个输出，并由n个状态变量描述，则其大小为n*p；
+ C-输出矩阵，如果系统具有p个输入和q个输出，并由n个状态变量描述，则其大小为q*n；
+ D-直达矩阵，如果系统具有p个输入和q个输出，并由n个状态变量描述，则其大小为q*p。
+ iu-索引，指定为一个整数。
+ order-二阶节系数中行的顺序，指定为以下值之一：
    + down- 按照最靠近单位圆的极点对节段进行排序，使得第一行包含这些极点
    + up-按照最远离单位圆的极点对节段进行排序，使得第一行包含这些极点
+ scale-增益和分子系数的缩放，指定为以下值之一：
    + 'none' — 不应用缩放
    + 'inf' — 应用无穷范数缩放
    + 'two' — 应用二范数缩放

### 输出参数
+ sos-二阶数字滤波器的系数，用L*6的矩阵表示，L代表二阶节数字滤波器的个数，比如两个二阶节数字滤波器参数形式如下：
$$
sos = \left[ \begin{matrix} b_{01} & b_{11}  & b_{21} & a_{01} & a_{11} & a_{21}\\ b_{02} & b_{12}  & b_{22} & a_{02} & a_{12} & a_{22} \end{matrix} \right]
$$
其中b<sub>01</sub>,b<sub>11</sub>,b<sub>21</sub>为第一个二阶节数字滤波器分子系数，a<sub>01</sub>,a<sub>11</sub>,a<sub>21</sub>为第一个二阶节数字滤波器分母系数。同理第二行为第二个二阶节数字滤波器的系数。
+ g-系统增益，一个标量。

## Python函数描述：ss2sos
函数来源：自定义
### Python中自定义的ss2sos函数
```python
from scipy.signal import ss2tf, tf2sos

def ss2sos(A, B, C, D):

    b, a = ss2tf(A, B, C, D)
    sos = tf2sos(b, a)
    
    return sos
```
### 输入参数
+ A-状态矩阵，大小为n*n，其中n是系统的状态数；
+ B-输入矩阵，大小为n×p，其中n是系统的状态数，p是输入的数量；
+ C-输出矩阵，大小为q×n，其中n是系统的状态数，q是输出的数量；
+ D-直达矩阵，大小为q×p，其中p是输入的数量，q是输出的数量。

### 输出参数
+ sos-二阶节系数矩阵，每一行都表示一个二阶节，每个二阶节由分子多项式的系数、分母多项式的系数描述，具有两个二阶节的数字滤波器sos具体形式如下：  
$$
sos = \left[ \begin{matrix} b_{01} & b_{11}  & b_{21} & a_{01} & a_{11} & a_{21}\\ b_{02} & b_{12}  & b_{22} & a_{02} & a_{12} & a_{22} \end{matrix} \right]
$$

### Python函数工作原理
1. 借助Python原有函数ss2tf将数字滤波器的空间状态形式转换为传递函数形式；
2. 借助Python原有函数tf2sos将数字滤波器的传递函数形式转换为二阶节系数形式。

### 注意事项
1. 函数使用只能是单输出系统，对于多输出系统应采用其他办法；
2. 由于运算方法的不同，Python和MATLAB对于同一个数字滤波器得出的二阶节系数sos可能有不同的表现形式，但都表示一个数字滤波器。

### 使用场景
将数字滤波器空间状态形式转换为二阶节参数形式。

### 改进建议
1. ss2sos函数只能是单输出系统，可加入对输入参数的检验功能保证函数能正确运行；
2. 可以考虑实现MATLAB中实现返回系统增益g的功能；
3. 在实际运算中，最终结果可能不是整数，因为运算方法、保留有效位数的不同，Python和MATLAB结果可能会有偏差，后续可能改进算法实现对应。

## Prompt 1 生成Python中ss2sos函数
在MATLAB中ss2sos函数代码如下：
```
% 创建ss空间状态矩阵
[A,B,C,D] = butter(5,0.2);
% 调用函数完成转换
[sos] = ss2sos(A,B,C,D);

disp(sos);
```
在Python中实现MATLAB对应过程
```python
import numpy as np
from scipy.signal import  butter, tf2ss, ss2tf, tf2sos

# 创建空间状态矩阵
b,a=butter(5,0.2)
A,B,C,D=tf2ss(b,a)

# 将状态空间形式的系统转换为传递函数形式
b, a = ss2tf(A, B, C, D)

# 将传递函数形式的系统转换为二阶节参数形式
sos = tf2sos(b, a)

print(sos)
```
### 定义Python中ss2sos函数
```python
from scipy.signal import ss2tf, tf2sos

def ss2sos(A, B, C, D):
    # 将状态空间形式的系统转换为传递函数形式
    b, a = ss2tf(A, B, C, D)
    # 将传递函数形式的系统转换为二阶节参数形式
    sos = tf2sos(b, a)
    return sos
```
## Prompt 2 使用自定义的Python中ss2sos函数与MATLAB示例对应
给出Python中待处理数据：
```Python
import numpy as np
from scipy.signal import  butter, tf2ss

# 创建空间状态矩阵
b,a=butter(5,0.2)
A,B,C,D=tf2ss(b,a)
```
Python中自定义的sos2ss函数如下：
```python
from scipy.signal import ss2tf, tf2sos

def ss2sos(A, B, C, D):

    b, a = ss2tf(A, B, C, D)
    sos = tf2sos(b, a)
    
    return sos
```
### 用Python完成示例对应
```python
import numpy as np
from scipy.signal import butter,tf2ss,ss2tf, tf2sos

def ss2sos(A, B, C, D):
    # 将状态空间形式的系统转换为传递函数形式
    b, a = ss2tf(A, B, C, D)
    # 将传递函数形式的系统转换为二阶节参数形式
    sos = tf2sos(b, a)
    return sos

b,a=butter(5,0.2)
A,B,C,D=tf2ss(b,a)

sos = ss2sos(A, B, C, D)
print("sos系数矩阵:", sos)
```
## Prompt 3 使用自定义的Python中ss2sos.py与MATLAB示例对应
MATLAB中处理的数据如下：
```
% 创建ss空间状态矩阵
[A,B,C,D] = butter(5,0.2);
% 调用函数完成转换
[sos] = ss2sos(A,B,C,D);

disp(sos);
```
Python中ss2sos.py文件如下：
```python
from scipy.signal import ss2tf, tf2sos

def ss2sos(A, B, C, D):
    
    b, a = ss2tf(A, B, C, D)
    sos = tf2sos(b, a)
    
    return sos
```
### 调用文件与MATLAB示例对应
```python
import numpy as np
from scipy.signal import butter, tf2ss
from ss2sos import ss2sos

#MATLAB中对应的空间状态矩阵
b,a=butter(5,0.2)
A,B,C,D=tf2ss(b,a)

# 调用自定义函数文件完成转换
sos = ss2sos(A, B, C, D)
print("sos系数矩阵:", sos)
```