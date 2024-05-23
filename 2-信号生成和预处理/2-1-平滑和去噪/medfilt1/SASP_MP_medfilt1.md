# **信号处理仿真与应用 - 信号处理工具箱 - 一维中值滤波**

## MATLAB函数描述：medfilt1

函数来源：[MATLAB medfilt1](https://ww2.mathworks.cn/help/signal/ref/medfilt1.html)

### 语法

```
y = medfilt1(x)
y = medfilt1(x,n)
y = medfilt1(x,n,blksz,dim)
y = medfilt1(x,n,[],dim)
y = medfilt1(___,nanflag,padding)
```

### 说明

y = medfilt1(x) 对输入向量 x 应用三阶一维中值滤波器。 该函数认为超出端点的信号为 0。 输出 y 与 x 具有相同的长度。

y = medfilt1(x,n,blksz,dim) 或 y = medfilt1(x,n,[],dim) 指定滤波器运行的维度 dim。 blksz 是向后兼容所必需的，但会被忽略。此语法指定填充，即在信号边缘执行的滤波类型。nanflag 和 padding 可以出现在函数调用中 x 之后的任何位置。

### 输入参数

- x - 输入信号

  矢量| 矩阵| N维阵列

  输入信号，指定为实值向量、矩阵或 N 维数组。

​		数据类型：single | double

- n - 过滤器阶数
  3（默认）| 正整数标量
  一维中值滤波器的阶数，指定为正整数标量。

  当n为奇数时，y(k)是x(k-(n-1)/2:k+(n-1)/2)的中位数。

  当 n 为偶数时，y(k) 是 x(k-n/2:k+(n/2)-1) 的中位数。 在本例中，该函数对数字进行排序，并取排序列表中两个中间元素的平均值。

  示例：如果 n = 11，则 y(k) 是 x(k-5:k+5) 的中值。

  示例：如果 n = 12，则 y(k) 是 x(k-6:k+5) 的中值。

  数据类型： double

- dim - 滤波器的维度
  正整数标量
  滤波器维度，指定为正整数标量。 默认情况下，该函数沿 x 的第一个非单一维度运行。 特别是，如果 x 是矩阵，则该函数会过滤其列。

  数据类型： double

- nanflag — NaN 条件

  includenan 默认 | omitnan
  NaN 条件，指定为 'includenan' 或 'omitnan'。

  'includenan' - 返回滤波后的信号，以便任何包含 NaN 的段的中值也为 NaN。

  'omitnan' - 返回滤波后的信号，以便任何包含 NaN 的段的中值是非 NaN 值的中值。 如果段中的所有元素均为 NaN，则结果为 NaN。

- padding - 端点过滤
  'zeropad'（默认）| 'truncate'
  端点过滤，指定为“zeropad”或“truncate”。

  'zeropad' - 认为信号在端点之外为零。

  'truncate' - 计算较小段到达信号边缘时的中值。

### 输出参量

- y - 滤波后的信号

  矢量| 矩阵| N维阵列

  滤波后的信号，以实值向量、矩阵或 N 维数组的形式返回。 y 与 x 大小相同。



## Python函数描述：scipy.signal.medfilt

函数来源：[Python medfilt1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html)

### 语法

```
scipy.signal.medfilt(volume, kernel_size=None)
```

对 N 维数组执行中值滤波器。

由*kernel_size*使用局部窗口大小将中值滤波器应用于输入数组 给出。该数组将自动用零填充。

### 参数
- **volume： array_like**
  N 维输入数组。
- **kernel_size：array_like, optional**
  一个标量或长度为N的列表，用于指定每个维度中中值滤波器窗口的大小。 kernel_size的元素应为奇数。 如果kernel_size是一个标量，那么这个标量将用作每个维度中的大小。 默认大小对于每个维度为3。

### 返回值
- **out**|ndarray

  与包含中值过滤结果的输入大小相同的数组。

### 注意事项
- scipy.ndimage.median_filter函数具有更高效的中值滤波实现，因此运行速度更快。
- 对于具有uint8、float32或float64数据类型的二维图像，专门的函数scipy.signal.medfilt2d可能更快。

## Prompt 1 ： 生成 Python 一维中值滤波器medfilt函数

参考下面MATLAB代码的medfilt1函数，生成一段python代码实现相同的功能。
```
% 噪声正弦波的 Savitzky-Golay 平滑
% 生成一个由嵌入高斯白噪声的 0.2 Hz 正弦波组成的信号，每秒采样五次，持续 200 秒。

dt = 1/5;
t = (0:dt:200-dt)';
x = 5*sin(2*pi*0.2*t) + randn(size(t));

% 使用sgolay平滑信号。使用 21 样本帧和四阶多项式。
order = 4;
framelen = 21;
b = sgolay(order,framelen);

% 通过将信号与 b 的中心行进行卷积来计算信号的稳态部分。
ycenter = conv(x,b((framelen+1)/2,:),'valid');
Compute the transients. Use the last rows of b for the startup and the first rows of b for the terminal.

% 计算瞬态。使用 b 的最后几行作为启动，使用 b 的第一行作为终端。
ybegin = b(end:-1:(framelen+3)/2,:) * x(framelen:-1:1);
yend = b((framelen-1)/2:-1:1,:) * x(end:-1:end-(framelen-1));

%连接瞬态部分和稳态部分以生成完整的平滑信号。绘制原始信号和 Savitzky-Golay 估计。
y = [ybegin; ycenter; yend];
plot([x y])
legend('Noisy Sinusoid','S-G smoothed sinusoid')
```

### LLM 生成 ： Python medfilt 函数

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    
    # Generate signal
    dt = 1/5
    t = np.arange(0, 200, dt)
    x = 5*np.sin(2*np.pi*0.2*t) + np.random.randn(len(t))
    
    # Smoothing using Savitzky-Golay filter
    order = 4
    framelen = 21
    ycenter = savgol_filter(x, framelen, order, mode='valid')
    
    # Compute transients
    ybegin = np.flipud(savgol_filter(x[:framelen], framelen, order, mode='mirror'))
    yend = savgol_filter(np.flipud(x[-framelen:]), framelen, order, mode='mirror')
    
    # Concatenate transients and steady-state portion
    y = np.concatenate((ybegin, ycenter, yend))
    
    # Plotting
    plt.plot(t, x, label='Noisy Sinusoid')
    plt.plot(t[:-framelen+1], y, label='S-G smoothed sinusoid')
    plt.legend()
    plt.show()


## Prompt 2 ： 解决模式参数报错问题

模式参数报错：ValueError: mode must be 'mirror', 'constant', 'nearest' 'wrap' or 'interp'.
### LLM 生成 ：

在 `savgol_filter` 函数中，模式参数 `mode` 的取值应该是 `['mirror', 'constant', 'nearest', 'wrap', 'interp']` 中的一个，而不是 `'mirror''constant', "nearest'"'wrap'`，因为这会导致 ValueError。让我们将模式参数修改为正确的形式，并重新运行代码： 

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Generate signal
dt = 1/5
t = np.arange(0, 200, dt)
x = 5*np.sin(2*np.pi*0.2*t) + np.random.randn(len(t))

# Smoothing using Savitzky-Golay filter
order = 4
framelen = 21
ycenter = savgol_filter(x, framelen, order, mode='mirror')  # 修改mode参数为正确的形式

# Compute transients
ybegin = np.flipud(savgol_filter(x[:framelen], framelen, order, mode='mirror'))
yend = savgol_filter(np.flipud(x[-framelen:]), framelen, order, mode='mirror')

# Concatenate transients and steady-state portion
y = np.concatenate((ybegin, ycenter, yend))

# Plotting
plt.plot(t, x, label='Noisy Sinusoid')
plt.plot(t[:-framelen+1], y, label='S-G smoothed sinusoid')
plt.legend()
plt.show()
```

## Prompt 3 ： 解决平滑后信号长度不一致导致的绘图报错问题

x和y信号长度不一致报错：ValueError:x and y must have same first dimension, but have shapes (980,)and (1042,)Output is truncated, View as a scrollable element or open in a text editor, Adjust cell output settings.


### LLM 生成 ： 

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Generate signal
dt = 1/5
t = np.arange(0, 200, dt)
x = 5*np.sin(2*np.pi*0.2*t) + np.random.randn(len(t))

# Smoothing using Savitzky-Golay filter
order = 4
framelen = 21
ycenter = savgol_filter(x, framelen, order, mode='mirror')  # 修改mode参数为正确的形式

# Compute transients
ybegin = np.flipud(savgol_filter(x[:framelen], framelen, order, mode='mirror'))
yend = savgol_filter(np.flipud(x[-framelen:]), framelen, order, mode='mirror')

# Concatenate transients and steady-state portion
y = np.concatenate((ybegin, ycenter, yend))

# Plotting
min_length = min(len(x), len(y))
plt.plot(t[:min_length], x[:min_length], label='Noisy Sinusoid')
plt.plot(t[:min_length], y[:min_length], label='S-G smoothed sinusoid')  # 修改平滑后信号的长度
plt.legend()
plt.show()
```



