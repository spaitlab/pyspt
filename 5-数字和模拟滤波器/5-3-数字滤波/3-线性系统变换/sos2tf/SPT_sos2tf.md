# 信号处理仿真和应用-数字和模拟滤波器-线性系统变换-sos2tf
## MATLAB函数描述：sos2tf
函数来源：[MATLAB sos2tf](https://ww2.mathworks.cn/help/signal/ref/sos2tf.html)
### 语法
[b,a]=sos2tf(sos)  
[b,a]=sos2tf(sos,g)
### 使用说明
[b,a]=sos2tf(sos)会返回输入二阶节形式数字滤波器的传递函数分子、分母系数  
[b,a]=sos2tf(sos,g)会返回具有增益为g的二阶节形式数字滤波器的分子、分母系数
### 输入参数
+ sos-二阶节数字滤波器的系数，用L*6的矩阵表示，L代表二阶节数字滤波器的个数，比如两个二阶节数字滤波器参数形式如下：
$$
sos = \left[ \begin{matrix} b_{01} & b_{11}  & b_{21} & a_{01} & a_{11} & a_{21}\\ b_{02} & b_{12}  & b_{22} & a_{02} & a_{12} & a_{22} \end{matrix} \right]
$$
其中b<sub>01</sub>,b<sub>11</sub>,b<sub>21</sub>为第一个二阶节数字滤波器分子系数，a<sub>01</sub>,a<sub>11</sub>,a<sub>21</sub>为第一个二阶节数字滤波器分母系数。同理第二行为第二个二阶节数字滤波器的系数。

+ g-数字滤波器系统增益，一个实标量。
### 输出参数
+ b-分子系数，返回一个代表传递函数分子系数的一维数组；
+ a-分母系数，返回一个代表传递函数分母系数的一维数组。


## Python函数描述：sos2tf
函数来源：自定义

### Python函数sos2tf自定义：
```python
from scipy.signal import sos2tf as scipy_sos2tf 

def sos2tf(sos, gain=1):

    b, a = scipy_sos2tf(sos)
    b *= gain
    return b, a
```
这是Python中自定义的sos2tf函数，在Python中原有sos2tf函数的基础上加入系统增益g输入参数，让其泛用性更高。

### 输入参数
+ sos：数字滤波器的二阶节参数，以L*6大小的数组输入；
+ g:系统增益，默认为1，增益为1时可以不输入。

### 输出参数
+ b-传递函数分子系数，返回一个代表传递函数分子系数的数组；
+ a-传递函数分母系数，返回一个代表传递函数分母系数的数组。

### Python函数工作原理
1. 借助原有函数sos2tf将二阶节参数转换为传递函数；
2. 将传递函数分子乘以增益g构成新的自定义sos2tf函数。

### 使用场景
将数字滤波器二阶节参数转化为传递函数形式

### 改进建议
1. 可以加入输入参数sos的大小检验，不符合输入时提示。

## Prompt 1 生成Python中sos2tf函数
在MATLAB中sos2tf函数代码如下：
```
% 定义一个二阶节参数矩阵
sos = [
        1, 1, 1, 1, 0, -1;   % 第一个二阶节参数
       -2, 3, 1, 1, 10, 1    % 第二个二阶节参数
      ];

% 将二阶节参数矩阵转换为传递函数的分子和分母形式
[b, a] = sos2tf(sos,2);

% 显示转换后的传递函数分子、分母系数
disp('传递函数分子系数:');
disp(b);
disp('传递函数分母系数:');
disp(a);
```
在Python中实现MATLAB中sos2tf过程
```Python
import numpy as np
from scipy.signal import sos2tf

# 给定二阶节系数
sos = np.array([
                [1, 1, 1, 1, 0, -1],
                [-2, 3, 1, 1, 10, 1]
                ])
# 增益因子
g = 2

b, a = sos2tf(sos)

b*=g
# 显示转换后的有理多项式系数
print("分子多项式系数:", b)
print("分母多项式系数:", a)
```

### 定义Python中sos2tf函数
```Python
from scipy.signal import sos2tf as scipy_sos2tf

def sos2tf(sos, gain=1):
    b, a = scipy_sos2tf(sos)
    # 将增益因子乘入分子系数
    b *= gain
    return b, a
```

## Prompt 2 使用自定义的Python中sos2tf函数与MATLAB示例对应
给出待处理数据：
```Python
# 给定二阶节系数
sos = np.array([
                [1, 1, 1, 1, 0, -1],
                [-2, 3, 1, 1, 10, 1]
                ])
# 增益因子
g = 2
```
Python自定义的sos2tf函数：
```Python
from scipy.signal import sos2tf as scipy_sos2tf

def sos2tf(sos, gain=1):
    b, a = scipy_sos2tf(sos)
    b *= gain
    return b, a
```
### 使用Python完成示例
```Python
import numpy as np
from scipy.signal import sos2tf as scipy_sos2tf

def sos2tf(sos, gain=1):

    b, a = scipy_sos2tf(sos)
    b *= gain
    return b, a

# 给定二阶节系数
sos = np.array([
                [1, 1, 1, 1, 0, -1],
                [-2, 3, 1, 1, 10, 1]
                ])
# 增益因子
g = 2

# 使用自定义的sos2tf函数将二阶节系数转换为有理多项式形式，并应用增益因子
b, a = sos2tf(sos, g)

print("分子多项式系数:", b)
print("分母多项式系数:", a)
```
## Prompt 3 使用自定义的Python中sos2tf.py与MATLAB示例对应
MATLAB中处理数据如下：
```
% 定义一个二阶节参数矩阵
sos = [
        1, 1, 1, 1, 0, -1;   % 第一个二阶节参数
       -2, 3, 1, 1, 10, 1;   % 第二个二阶节参数
      ];

% 将二阶节参数矩阵转换为传递函数的分子和分母形式
[b, a] = sos2tf(sos,2);

% 显示转换后的传递函数分子、分母系数
disp('传递函数分子系数:');
disp(b);
disp('传递函数分母系数:');
disp(a);
```
Python中sos2tf.py文件如下：
```Python
from scipy.signal import sos2tf as scipy_sos2tf

def sos2tf(sos, gain=1):
    b, a = scipy_sos2tf(sos)
    b *= gain
    return b, a
```
Python中通过调用sos2tf.py实现程序
```Python
from sos2tf import sos2tf
import numpy as np
# 给定二阶节系数
sos = np.array([
                [1, 1, 1, 1, 0, -1],
                [-2, 3, 1, 1, 10, 1]
                ])
# 增益因子
g = 2

# 使用自定义的sos2tf函数将二阶节系数转换为有理多项式形式，并应用增益因子
b, a = sos2tf(sos, g)

print("分子多项式系数:", b)
print("分母多项式系数:", a)
```



