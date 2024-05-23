# **信号处理仿真与应用 - 信号处理工具箱 - Savitzky-Golay 滤波器设计**

## MATLAB函数描述：sgolay

函数来源：[MATLAB sgolay](https://ww2.mathworks.cn/help/signal/ref/sgolay.html)

### 语法

```
b = sgolay(order,framelen)
b = sgolay(order,framelen,weights)
[b,g] = sgolay(___)
```

### 说明

b = sgolay(order,framelen) 使用多项式阶数order和帧长framelen设计Savitzky-Golay FIR平滑滤波器。

b = sgolay(order,framelen,weights) 指定一个权重向量weights，其中包含在最小二乘最小化过程中使用的实数、正值权重。

示例 [b,g] = sgolay(___) 返回微分滤波器矩阵g。您可以将这些输出参数与任何先前的输入语法一起使用。

### 输入参数

- order — Polynomial order

  positive integer

  指定为正整数，阶数的值必须小于帧长度。如果阶数 = 帧长度 - 1，则设计的滤波器不会产生平滑效果。

- framelen — Frame length

  positive odd integer

  指定为正奇整数。帧长度的值必须大于阶数。

- weights — Weighting vector
  real positive vector

  指定为实数正向量。加权向量的长度与帧长度相同，并用于执行最小二乘法最小化。

### 输出参量

- b — Time-varying FIR filter coefficients 

  matrix 

  时间变化的FIR滤波器系数，以framelen乘以framelen的矩阵形式指定。在平滑滤波器实现中（例如sgolayfilt），在启动瞬态期间，将应用最后的（framelen-1）/2行（每行一个FIR滤波器）到信号上，而在终端瞬态期间将应用前（framelen-1）/2行到信号上。中间行将在稳态期间应用到信号上。

- g — Matrix of differentiation filters 

  matrix 

  微分滤波器矩阵，以矩阵形式指定。

  g的每一列是一个用于p-1阶导数的微分滤波器，其中p是列索引。给定长度为framelen的信号x，您可以从xp((framelen+1)/2) = (阶乘(p)) * g(:,p+1)' * x中找到其中值的第p阶导数xp的估计值。



## Python函数描述：scipy.signal.savgol_filter

函数来源：[Python savgol_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)

### 语法

```
scipy.signal.savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
```

将 Savitzky-Golay 过滤器应用于数组。 这是一个一维滤波器。如果 x 的维度大于 1，则 axis 确定应用过滤器的轴。

### 参数
- **x - array_like**

  要过滤的数据。 如果 *x* 不是单精度或双精度浮点数组，则在过滤之前它将转换为类型“numpy.float64”。

- **window_length - int**

  滤波器窗口的长度（即系数的数量）。 如果 *mode* 为 ‘interp’，则 *window_length* 必须小于或等于 *x* 的大小。

- **polyorder** **- int**

  用于拟合样本的多项式的阶数。 *polyorder* 必须小于*window_length*。

- **deriv** **- int, optional**

  要计算的导数的阶数。 这必须是一个非负整数。 默认为0，表示对数据进行不区分的过滤。

- **delta** **- float, optional**

  将应用过滤器的样本的间距。 仅当 deriv > 0 时才使用。默认值为 1.0。

- **axis** **- int, optional**

  数组 *x* 的轴，滤波器将沿着该轴应用。 默认值为-1。

- **mode** **- str, optional**

  必须是‘mirror’, ‘constant’, ‘nearest’, ‘wrap’ 或 ‘interp’. 。决定用于应用滤波器的填充信号的扩展类型。当模式为‘constant’时，填充值由cval给出。当选择“interp”模式（默认情况下），不使用扩展。相反，将最后的window_length个值拟合为degree polyorder多项式，然后使用该多项式来评估最后的window_length // 2个输出值。

- **cval** **- scalar, optional**

  如果模式是'constant'，则填充超出输入边缘的值。默认值为0.0。 

### 返回值
- **y** **- ndarray, same shape as *x***

  滤波器数据

### 注意事项
- ‘mirror’:
  以相反的顺序重复边缘处的值。 最接近边缘的值不包括在内。
- ‘nearest’:
  扩展名包含最接近的输入值。
- ‘constant’:
  扩展包含 cval 参数给出的值。
- ‘wrap’:
  扩展名包含数组另一端的值。

## Prompt 1 ： 生成 Python Savitzky-Golay 滤波器设计案例

参考下面的MATLAB代码，生成python代码实现相同的功能

```matlab
% 生成以 100 Hz 采样 1 秒的正弦信号，添加较高频率的正弦波来模拟噪声。
fs = 100;
t = 0:1/fs:1;
x = sin(2*pi*t*3)+0.25*sin(2*pi*t*40);
% 使用 10 阶中值滤波器来平滑信号。绘制结果。
y = medfilt1(x,10);
plot(t,x,t,y)
legend('Original','Filtered')
legend('boxoff')
```

### LLM 生成 ： Python Savitzky-Golay 滤波器设计

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

fs = 100
t = np.arange(0, 1, 1/fs)
x = np.sin(2*np.pi*t*3) + 0.25*np.sin(2*np.pi*t*40)

# Apply 10th-order median filter to smooth the signal
y = medfilt(x, kernel_size=10)

plt.plot(t, x, label='Original')
plt.plot(t, y, label='Filtered')
plt.legend()
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
plt.show()
```


## Prompt 2 ：纠正 Python Savitzky-Golay 滤波器设计传参偶数无法处理问题

报错信息：ValueError:Each element of kernel size should be odd.

### LLM 生成 ： 

因为 `kernel_size` 参数必须是奇数，而案例提供的是一个偶数。让我们将 `kernel_size` 参数改为奇数，然后再运行代码试试。

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

fs = 100
t = np.arange(0, 1, 1/fs)
x = np.sin(2*np.pi*t*3) + 0.25*np.sin(2*np.pi*t*40)

# Apply 10th-order median filter to smooth the signal
y = medfilt(x, kernel_size=11)  # Changed kernel_size to an odd number

plt.plot(t, x, label='Original')
plt.plot(t, y, label='Filtered')
plt.legend()
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
plt.show()
```

