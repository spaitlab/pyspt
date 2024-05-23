# 信号处理仿真和应用-数字和模拟滤波器-线性系统变换-ss
## MATLAB函数描述：ss
函数来源：[[MATLAB ss](https://ww2.mathworks.cn/help/signal/ref/ss.html?s_tid=doc_ta)]
### 语法
[A,B,C,D]=ss(d) 
### 使用说明
[A,B,C,D]=ss(d) 会返回设计的数字滤波器的空间状态形式
### 输入参数
+ d-使用MATLAB内部函数[designfilt](https://ww2.mathworks.cn/help/signal/ref/designfilt.html?s_tid=doc_ta)生成的数字滤波器
### 输出参数
+ A-状态矩阵，指定为一个矩阵。如果系统具有p个输入和q个输出，并由n个状态变量描述，则状态矩阵大小为n*n；
+ B-输入矩阵，如果系统具有p个输入和q个输出，并由n个状态变量描述，则其大小为n*p；
+ C-输出矩阵，如果系统具有p个输入和q个输出，并由n个状态变量描述，则其大小为q*n；
+ D-直达矩阵，如果系统具有p个输入和q个输出，并由n个状态变量描述，则其大小为q*p。

## Python函数描述：tf2ss
函数来源：Python中scipy.signal模块的[tf2ss](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2ss.html#scipy.signal.tf2ss)函数
### 语法
[A,B,C,D]=tf2ss(b,a) 
### 使用说明
[A,B,C,D]=tf2ss(b,a) 输入数字滤波器传递函数的分子、分母系数，返回数字滤波器的空间状态形式
### 输入参数
+ b-数字滤波器传递函数的分子系数  
+ a-数字滤波器传递函数的分母系数
### 输出参数
+ A-状态矩阵，指定为一个矩阵。如果系统具有p个输入和q个输出，并由n个状态变量描述，则状态矩阵大小为n*n；
+ B-输入矩阵，如果系统具有p个输入和q个输出，并由n个状态变量描述，则其大小为n*p；
+ C-输出矩阵，如果系统具有p个输入和q个输出，并由n个状态变量描述，则其大小为q*n；
+ D-直达矩阵，如果系统具有p个输入和q个输出，并由n个状态变量描述，则其大小为q*p。

### Python函数工作原理
1. 借助Python原有函数[iirfilter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirfilter.html#scipy.signal.iirfilter)、[butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter)等Python函数设计数字滤波器时可以返回传递函数的分子、分母系数；
2. 使用Python中内部函数tf2ss可以将传递函数分子、分母系数转化为状态空间形式。

### 使用场景
在已知设计的数字滤波器的传递函数分子、分母系数的情况下将其转化为状态空间形式

### 改进建议
1. tf2ss的使用需要知道数字滤波器传递函数分子、分母系数，后续可以改进实现对于其它不返回传递函数系数的数字滤波器的应用；
2. 还可以通过获取数字滤波器零点、极点、增益和二阶节系数，再将其转换为状态空间形式，实现函数在多种情况下的使用。

## Prompt 1 使用Python函数中tf2ss完成MATLAB中示例对应
在MATLAB中ss函数代码如下：
```
% 设计低通IIR滤波器
d = designfilt('lowpassiir', 'FilterOrder', 6, 'PassbandFrequency', 0.2);

% 将数字滤波器转换为状态空间表示
[A, B, C, D] = ss(d);

disp('状态转移矩阵 A:');
disp(A);
disp('输入矩阵 B:');
disp(B);
disp('输出矩阵 C:');
disp(C);
disp('直达矩阵 D:');
disp(D);
```
在Python中实现MATLAB中ss函数过程
```Python
import numpy as np
from scipy.signal import iirfilter, tf2ss

# 设计 IIR 低通滤波器
order = 6
cutoff = 0.2
b, a = iirfilter(order, cutoff, btype='low', analog=False, ftype='butter')

# 将滤波器转换为状态空间表示
A, B, C, D = tf2ss(b, a)

print("A矩阵:\n", A)
print("B矩阵:\n", B)
print("C矩阵:\n", C)
print("D矩阵:\n", D)
```