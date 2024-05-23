# 信号处理仿真与应用 - 信号生成和预处理 - 波形生成

## MATLAB函数描述：rectpuls

函数来源：[MATLAB rectpuls ](https://ww2.mathworks.cn/help/signal/ref/rectpuls.html)

### 语法

y = rectpuls(t)
y = rectpuls(t,w)

### 说明

y = rectpuls(t)返回一个以数组 t 中指示的采样时间采样的连续非周期性单位高度矩形脉冲，该矩形脉冲以 t = 0 为中心。

y = rectpuls(t,w)生成一个宽度为 w的矩形。

### 输入参数

t — 采样时间
向量

单位矩形脉冲的采样时间，指定为向量。

数据类型: single | double

w — 矩形宽度
1 (默认) | 正数

矩形宽度，指定为正数。

### 输出参数

y — 矩形脉冲
向量

单位振幅的矩形脉冲，以向量形式返回。



## Python函数描述：rectpuls

函数来源：自定义

```python
import numpy as np

def rectpuls(t, width):
    """
    生成矩形脉冲信号

    参数：
    t : array_like
        时间向量
    width : float
        脉冲宽度

    返回：
    x : ndarray
        生成的矩形脉冲信号
    """
    x = np.zeros_like(t)
    x[np.abs(t) <= width / 2] = 1
    return x

```



## Prompt 1 ： 生成 Python rectpuls 函数

参考下面MATLAB代码的rectpuls函数

```
fs = 10e3;
t = -0.1:1/fs:0.1;

w = 20e-3;

x = rectpuls(t,w);
```

完成Python语言rectpuls函数的编写，便于类似应用调用。

### LLM 生成 ： Python rectpuls 函数

你想要实现的是一个在Python中类似于MATLAB的rectpuls函数的功能，这个函数可以生成矩形脉冲信号。在Python中，你可以使用NumPy库来实现类似的功能。下面是一个简单的实现：

```python
import numpy as np
import matplotlib.pyplot as plt
def rectpuls(t, width):
    """
    生成矩形脉冲信号

    参数：
    t : array_like
        时间向量
    width : float
        脉冲宽度

    返回：
    x : ndarray
        生成的矩形脉冲信号
    """
    x = np.zeros_like(t)
    x[np.abs(t) <= width / 2] = 1
    return x

# 示例用法
fs = 10e3
t = np.arange(-0.1, 0.1, 1/fs)
w = 20e-3

x = rectpuls(t, w)
plt.plot(t, x)
plt.show()
```

这个函数首先创建一个和输入时间向量`t`相同大小的零数组，然后在时间范围内宽度为`width`的区域设置为1，其他区域保持为0。最后返回生成的矩形脉冲信号。

## Prompt 2 ： 应用生成的 Python rectpuls 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：

```
fs = 10e3;
t = -0.1:1/fs:0.1;

w = 20e-3;

x = rectpuls(t,w);

tpast = -45e-3;
xpast = rectpuls(t-tpast,w);

tfutr = 60e-3;
xfutr = rectpuls(t-tfutr,w/2);

plot(t,x,t,xpast,t,xfutr)
ylim([-0.2 1.2])
```

要求画出图像中不同的波形颜色不同

### LLM 生成 ：

```
import numpy as np
import matplotlib.pyplot as plt

def rectpuls(t, width):
    """
    生成矩形脉冲信号

    参数：
    t : array_like
        时间向量
    width : float
        脉冲宽度

    返回：
    x : ndarray
        生成的矩形脉冲信号
    """
    x = np.zeros_like(t)
    x[np.abs(t) <= width / 2] = 1
    return x

# 生成时间向量
fs = 10e3
t = np.arange(-0.1, 0.1, 1/fs)

# 定义脉冲宽度
w = 20e-3

# 生成矩形脉冲信号
x = rectpuls(t, w)

# 定义过去和未来的时间偏移
tpast = -45e-3
tfutr = 60e-3

# 生成过去和未来的矩形脉冲信号
xpast = rectpuls(t - tpast, w)
xfutr = rectpuls(t - tfutr, w/2)

# 绘制图像
plt.plot(t, x, label='Current', color='blue')
plt.plot(t, xpast, label='Past', color='red')
plt.plot(t, xfutr, label='Future', color='green')
plt.ylim([-0.2, 1.2])
plt.legend()
plt.show()

```

