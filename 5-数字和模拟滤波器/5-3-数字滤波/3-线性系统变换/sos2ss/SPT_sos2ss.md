# 信号处理仿真和应用-数字和模拟滤波器-线性系统变换-sos2ss
## MATLAB函数描述：sos2ss
函数来源：[[MATLAB sos2ss](https://www.mathworks.com/help/signal/ref/sos2ss.html?s_tid=doc_ta)]
### 语法
[A,B,C,D]=sos2ss(sos)  
[A,B,C,D]=sos2ss(sos,g)
### 使用说明
[A,B,C,D]=sos2ss(sos)会返回输入二阶节形式数字滤波器的状态空间形式  
[A,B,C,D]=sos2ss(sos,g)会返回具有增益为g的二阶节形式数字滤波器的状态空间形式
### 输入参数
+ sos-二阶数字滤波器的系数，用L*6的矩阵表示，L代表二阶节数字滤波器的个数，比如两个二阶节数字滤波器参数形式如下：
$$
sos = \left[ \begin{matrix} b_{01} & b_{11}  & b_{21} & a_{01} & a_{11} & a_{21}\\ b_{02} & b_{12}  & b_{22} & a_{02} & a_{12} & a_{22} \end{matrix} \right]
$$
其中b<sub>01</sub>,b<sub>11</sub>,b<sub>21</sub>为第一个二阶节数字滤波器分子系数，a<sub>01</sub>,a<sub>11</sub>,a<sub>21</sub>为第一个二阶节数字滤波器分母系数。同理第二行为第二个二阶节数字滤波器的系数。

+ g-数字滤波器系统增益，默认为1，增益为1时可以不输入。
### 输出参数
+ A-状态矩阵，返回一个2L*2L大小的矩阵，描述系统中各个状态量之间的变化关系；
+ B-输入矩阵，返回一个1*L大小的矩阵，描述系统对外部输入信号的响应方式；
+ C-输出矩阵，返回一个L*1大小的矩阵，描述状态如何转化为输出信号；
+ D-直达矩阵，返回一个数值。


## Python函数描述：sos2ss
函数来源：自定义

### Python函数sos2ss自定义：
```python
from scipy.signal import sos2tf, tf2ss

def sos2ss(sos, g=1):

    b, a = sos2tf(sos)
    b = [g * coef for coef in b]

    A, B, C, D = tf2ss(b, a)
    return A, B, C, D
```
这段函数自定义了MATLAB中sos2ss函数对应的Python函数，将二阶节参数转化为传递函数，再转化为状态空间方程形式。

### 输入参数
+ sos：数字滤波器的二阶节参数，以L*6大小的数组输入，L为数字滤波器二阶节个数；
+ g:系统增益，默认为1。

### 输出参数
+ A-状态矩阵，返回一个2L*2L大小的矩阵；
+ B-输入矩阵，返回一个1*L大小的矩阵；
+ C-输出矩阵，返回一个L*1大小的矩阵；
+ D-直达矩阵，返回一个标量。

### Python函数工作原理
1. 先将每个二阶节参数用sos2tf转换为传递函数；
2. 再将传递函数分子系数乘以系统增益得到总的传递函数；
3. 最后借助函数tf2ss将传递函数转换为状态空间方程。

### 使用场景
状态空间方程利于系统的设计和优化，评估系统性能。

### 改进建议
1. 可以增加增益g的输入个数，用来实现多个不同增益数字滤波器的整合集成；
2. 提高计算性能，减少大规模运算时间。

## Prompt 1 生成Python中sos2ss函数

MATLAB中sos2ss的实现：
```
sos = [1, 1, 1, 1, 0, -1;
      -2, 3, 1, 1, 10, 1];

[A, B, C, D] = sos2ss(sos,2);

disp('状态矩阵 A:');
disp(A);
disp('输入矩阵 B:');
disp(B);
disp('输出矩阵 C:');
disp(C);
disp('直达矩阵 D:');
disp(D);
```
使用Python完成MATLAB中过程：
```python
import numpy as np
from scipy.signal import sos2tf, tf2ss

sos = np.array([
                 [1, 1, 1, 1, 0, -1],
                 [-2, 3, 1, 1, 10, 1]
                ])

b, a = sos2tf(sos)

# 定义系统增益因子
g = 2

b*=g

A, B, C, D = tf2ss(b, a)

print("A 矩阵:")
print(A)
print("B 矩阵:")
print(B)
print("C 矩阵:")
print(C)
print("D 矩阵:")
print(D)
```
### 定义Python中sos2ss函数
```python
from scipy.signal import sos2tf, tf2ss

def sos2ss(sos, g=1):
    # 将SOS表示的数字滤波器转换为传递函数的分子和分母形式
    b, a = sos2tf(sos)

    # 将传递函数乘以增益因子
    b = [g * coef for coef in b]

    # 使用传递函数转换为状态空间方程
    A, B, C, D = tf2ss(b, a)
    return A, B, C, D
```
## Prompt 2 使用自定义的Python中sos2ss函数与MATLAB示例对应
给出Python中待处理数据：
```Python
# 定义数字滤波器的二阶节参数
sos = np.array([
                 [1, 1, 1, 1, 0, -1],
                 [-2, 3, 1, 1, 10, 1]
                ])

# 系统增益为2
g = 2
```
Python中自定义的sos2ss函数如下：
```python
from scipy.signal import sos2tf, tf2ss

def sos2ss(sos, g=1):

    b, a = sos2tf(sos)
    b = [g * coef for coef in b]

    A, B, C, D = tf2ss(b, a)
    return A, B, C, D
```
### 用Python完成示例对应
```python
import numpy as np
from scipy.signal import sos2tf, tf2ss

def sos2ss(sos, g=1):
    # 将SOS表示的数字滤波器转换为传递函数的分子和分母形式
    b, a = sos2tf(sos)

    # 将传递函数乘以增益因子
    b = [g * coef for coef in b]

    # 使用传递函数转换为状态空间方程
    A, B, C, D = tf2ss(b, a)
    return A, B, C, D

sos = np.array([
                 [1, 1, 1, 1, 0, -1],
                 [-2, 3, 1, 1, 10, 1]
                ])

# 定义系统增益因子
g = 2

# 将二阶节系数和增益作为输入参数
A, B, C, D = sos2ss(sos, g)

print("A 矩阵:")
print(A)
print("B 矩阵:")
print(B)
print("C 矩阵:")
print(C)
print("D 矩阵:")
print(D)
```
## Prompt 3 使用自定义的Python中sos2ss.py与MATLAB示例对应
MATLAB中处理的二阶节数据如下：
```
% 定义数字滤波器的二阶节参数
sos = [1, 1, 1, 1, 0, -1;
      -2, 3, 1, 1, 10, 1];

% 使用sos2ss将二阶节参数转换为状态空间形式
[A, B, C, D] = sos2ss(sos,3);

% 输出状态空间模型的系数矩阵
disp('状态矩阵 A:');
disp(A);
disp('输入矩阵 B:');
disp(B);
disp('输出矩阵 C:');
disp(C);
disp('直达矩阵 D:');
disp(D);
```
Python中sos2ss.py文件如下：
```python
from scipy.signal import sos2tf, tf2ss

def sos2ss(sos, g=1):
    # 将SOS表示的数字滤波器转换为传递函数的分子和分母形式
    b, a = sos2tf(sos)

    # 将传递函数乘以增益因子
    b = [g * coef for coef in b]

    # 使用传递函数转换为状态空间方程
    A, B, C, D = tf2ss(b, a)
    return A, B, C, D
```
### 调用文件与MATLAB示例对应
```python
from sos2ss import sos2ss
import numpy as np

sos = np.array([
                 [1, 1, 1, 1, 0, -1],
                 [-2, 3, 1, 1, 10, 1]
                ])

# 定义系统增益因子
g=2

# 将二阶节系数和增益作为输入参数
A, B, C, D = sos2ss(sos,g)

print("A 矩阵:")
print(A)
print("B 矩阵:")
print(B)
print("C 矩阵:")
print(C)
print("D 矩阵:")
print(D)
```

