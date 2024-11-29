# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波器设计

## MATLAB函数描述：sgolay

Savitzky-Golay 滤波器设计

函数来源：[MATLAB sgolay](https://ww2.mathworks.cn/help/signal/ref/sgolay.html)

### 语法

b = sgolay(order,framelen)
b = sgolay(order,framelen,weights)
[b,g] = sgolay(___)

### 说明

b = sgolay(order,framelen)设计了一个多项式阶 order 和帧长 framelen 的 Savitzky-Golay FIR平滑滤波器。

b = sgolay(order, framelen,weights)指定一个权重向量weights，其中包含在最小二乘最小化过程中使用的实数、正值权重。

[b,g] = sgolay(___)返回差分滤波器的矩阵 g 。您可以将这些输出参数与前面的任何输入语法一起使用。

### 输入参数

- order — 多项式阶数
  
    正整数
  
    多项式阶数，被指定为正指数。order的值必须小于framelen。如果order = framelen - 1，则设计的滤波器不产生平滑。

- framelen - 帧长度
  
    正奇整数

    帧长度，指定为正奇数。framelen的值必须大于order。


- weights - 权重向量

    实正向量

    加权向量，指定为实正向量。权重向量具有与framelen相同的长度，并用于执行最小二乘最小化。

### 输出参量

- b - 时变FIR滤波器系数

    矩阵

    时变FIR滤波器系数，指定为framelen*framelen的矩阵。在平滑滤波器实现(例如，sgolayfilt)中，在启动瞬态期间将最后(framelen-1)/2行(每个FIR滤波器)应用于信号，并在终端瞬态期间将第一(framelen-1)/2行应用于信号。中心行应用于稳态下的信号。

- g - 差分滤波器矩阵

    矩阵

    矩阵的微分滤波器，指定为一个矩阵。 g 的每一列是p-1阶导数的微分滤波器，其中p是列索引。给定一个长度为“framelen”的信号x，你可以通过xp((framelen+1)/2) = (factorial(p)) * g(:,p+1)' * x来估算其中间值的p阶导数xp。

### 算法

Savitzky-Golay平滑滤波器(也称为数字平滑多项式滤波器或最小二乘平滑滤波器)通常用于“平滑”频率跨度(无噪声)较大的噪声信号。在这种类型的应用中，Savitzky-Golay平滑滤波器比标准平均FIR滤波器表现得更好，后者倾向于滤除信号高频内容的很大一部分以及噪声。

您可以实现数据平滑来测量缓慢变化且被随机噪声破坏的变量。由于附近的点测量的基本值几乎相同，因此可以用周围数据点的局部平均值替换每个数据点。Savitzky-Golay滤波器是最优的，因为它们在对每一帧噪声数据拟合多项式时，使最小二乘误差最小化。

## Python函数描述：savgol_filter

函数来源：

[scipy.signal.savgol_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)

### 语法


### 函数savgol_filter

scipy.signal.savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

对数组应用Savitzky-Golay过滤器。

这是一个一维滤波器。如果x的维度大于1，则axis确定应用过滤器的轴。

#### 参数

- `x`: (array_like) 

待过滤的数据。如果x不是单精度或双精度浮点数数组，它将在过滤前被转换为numpy.float64类型。

- `window_length`:  (int)   

滤波器窗口的长度（即系数的数量）。如果mode为'interp'，则window_length必须小于或等于x的大小。

- `polyorder`: (int)

用于拟合样本的多项式的阶数。polyorder必须小于window_length。
  
- `deriv`: (int) 可选择的

要计算的导数的阶数。这必须是非负整数。默认值为0，表示对数据进行过滤而不进行微分。

- `delta`: (float) 可选择的

应用滤波器的样本间距。这仅在deriv > 0时使用。默认值为1.0。

- `axis`: (int) 可选择的

应用滤波器的数组x的轴。默认值为-1。

- `mode`: (str) 可选择的

必须是'mirror'、'constant'、'nearest'、'wrap'或'interp'。这决定了应用于填充信号以应用滤波器的扩展类型。当mode为'constant'时，填充值由cval给出。有关'mirror'、'constant'、'wrap'和'nearest'的更多详细信息，请参阅说明部分。当选择'interp'模式（默认值）时，不使用扩展。相反，将拟合一个degree polyorder多项式到边缘的最后window_length个值，并使用此多项式来评估最后window_length // 2个输出值。

- `cval`: (scalar) 可选择的

如果mode为'constant'，则用于填充输入边缘之外的值。默认值为0.0。

#### 返回值

- `y`: (ndarray) 与`x`有相同的形状。

#### 注释

关于mode选项的详细解释：

- 'mirror'：

    以相反的顺序重复边缘处的值。最接近边缘的值不包括在内。

- 'nearest'：

    扩展部分包含最近的输入值。

- 'constant'：

    扩展部分包含由cval参数指定的值。

- 'wrap'：

    扩展部分包含从数组另一端取得的值。

## Prompt 1 ： 应用 Python savgol_filter 函数完成 Matlab 函数示例——噪声正弦波的Savitzky-Golay平滑

参考下面MATLAB代码sgolay函数例子：
```
dt = 1/5;
t = (0:dt:200-dt)';

x = 5*sin(2*pi*0.2*t) + randn(size(t));
order = 4;
framelen = 21;

b = sgolay(order,framelen);
ycenter = conv(x,b((framelen+1)/2,:),'valid');
ybegin = b(end:-1:(framelen+3)/2,:) * x(framelen:-1:1);
yend = b((framelen-1)/2:-1:1,:) * x(end:-1:end-(framelen-1));
y = [ybegin; ycenter; yend];
plot([x y])
legend('Noisy Sinusoid','S-G smoothed sinusoid')
```

和Python中的函数:scipy.signal.savgol_filter源码：
```
def savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0,
                  axis=-1, mode='interp', cval=0.0):
    """ Apply a Savitzky-Golay filter to an array.

    This is a 1-D filter. If `x`  has dimension greater than 1, `axis`
    determines the axis along which the filter is applied.

    Parameters
    ----------
    x : array_like
        The data to be filtered. If `x` is not a single or double precision
        floating point array, it will be converted to type ``numpy.float64``
        before filtering.
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
        If `mode` is 'interp', `window_length` must be less than or equal
        to the size of `x`.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0. Default is 1.0.
    axis : int, optional
        The axis of the array `x` along which the filter is to be applied.
        Default is -1.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : scalar, optional
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.

    Returns
    -------
    y : ndarray, same shape as `x`
        The filtered data.

    See Also
    --------
    savgol_coeffs

    Notes
    -----
    Details on the `mode` options:

        'mirror':
            Repeats the values at the edges in reverse order. The value
            closest to the edge is not included.
        'nearest':
            The extension contains the nearest input value.
        'constant':
            The extension contains the value given by the `cval` argument.
        'wrap':
            The extension contains the values from the other end of the array.

    For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8], and
    `window_length` is 7, the following shows the extended data for
    the various `mode` options (assuming `cval` is 0)::

        mode       |   Ext   |         Input          |   Ext
        -----------+---------+------------------------+---------
        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
        'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import savgol_filter
    >>> np.set_printoptions(precision=2)  # For compact display.
    >>> x = np.array([2, 2, 5, 2, 1, 0, 1, 4, 9])

    Filter with a window length of 5 and a degree 2 polynomial.  Use
    the defaults for all other parameters.

    >>> savgol_filter(x, 5, 2)
    array([1.66, 3.17, 3.54, 2.86, 0.66, 0.17, 1.  , 4.  , 9.  ])

    Note that the last five values in x are samples of a parabola, so
    when mode='interp' (the default) is used with polyorder=2, the last
    three values are unchanged. Compare that to, for example,
    `mode='nearest'`:

    >>> savgol_filter(x, 5, 2, mode='nearest')
    array([1.74, 3.03, 3.54, 2.86, 0.66, 0.17, 1.  , 4.6 , 7.97])

    """
    if mode not in ["mirror", "constant", "nearest", "interp", "wrap"]:
        raise ValueError("mode must be 'mirror', 'constant', 'nearest' "
                         "'wrap' or 'interp'.")

    x = np.asarray(x)
    # Ensure that x is either single or double precision floating point.
    if x.dtype != np.float64 and x.dtype != np.float32:
        x = x.astype(np.float64)

    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

    if mode == "interp":
        if window_length > x.shape[axis]:
            raise ValueError("If mode is 'interp', window_length must be less "
                             "than or equal to the size of x.")

        # Do not pad. Instead, for the elements within `window_length // 2`
        # of the ends of the sequence, use the polynomial that is fitted to
        # the last `window_length` elements.
        y = convolve1d(x, coeffs, axis=axis, mode="constant")
        _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y)
    else:
        # Any mode other than 'interp' is passed on to ndimage.convolve1d.
        y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=cval)

    return y
```
请使用Python的scipy.signal库中的savgol_filter函数，完成Matlab软件sgolay函数的示例，完成MATLAB函数示例到Python程序示例的对应。

### LLM 生成 ： Python savgol_filter 函数示例

以下是使用Python中的scipy.signal库中的savgol_filter函数来完成MATLAB软件sgolay函数示例的对应Python程序示例：

#### Python
```
import numpy as np
from scipy.signal import savgol_filter

dt = 1/5
t = np.arange(0, 200, dt)

x = 5*np.sin(2*np.pi*0.2*t) + np.random.randn(len(t))
order = 4
framelen = 21

ycenter = savgol_filter(x, framelen, order)[(framelen+1)//2:len(x)-(framelen-1)//2]
ybegin = savgol_filter(x, framelen, order)[:framelen]
yend = savgol_filter(x, framelen, order)[len(x)-framelen:]
y = np.concatenate((ybegin, ycenter, yend))

import matplotlib.pyplot as plt
plt.plot(t, x, label='Noisy Sinusoid')
plt.plot(t, y, label='S-G smoothed sinusoid')
plt.legend()
plt.show()
```
这段Python代码模拟了MATLAB软件sgolay函数的示例，并使用了Python中的scipy.signal库中的savgol_filter函数来实现相同的功能。
	
## Prompt 2 ： 应用 Python savgol_filter 函数完成 Matlab 函数示例—— Savitzky-Golay 微分

参考下面MATLAB代码sgolay函数例子：
```
dt = 0.25;
t = (0:dt:20-1)';

x = 5*sin(2*pi*0.2*t)+0.5*randn(size(t));
[b,g] = sgolay(5,25);

dx = zeros(length(x),4);
for p = 0:3
  dx(:,p+1) = conv(x, factorial(p)/(-dt)^p * g(:,p+1), 'same');
end
plot(x,'.-')
hold on
plot(dx)
hold off

legend('x','x (smoothed)','x''','x''''', 'x''''''')
title('Savitzky-Golay Derivative Estimates')
```

和Python中的函数:scipy.signal.savgol_filter源码：
```
def savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0,
                  axis=-1, mode='interp', cval=0.0):
    """ Apply a Savitzky-Golay filter to an array.

    This is a 1-D filter. If `x`  has dimension greater than 1, `axis`
    determines the axis along which the filter is applied.

    Parameters
    ----------
    x : array_like
        The data to be filtered. If `x` is not a single or double precision
        floating point array, it will be converted to type ``numpy.float64``
        before filtering.
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
        If `mode` is 'interp', `window_length` must be less than or equal
        to the size of `x`.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0. Default is 1.0.
    axis : int, optional
        The axis of the array `x` along which the filter is to be applied.
        Default is -1.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : scalar, optional
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.

    Returns
    -------
    y : ndarray, same shape as `x`
        The filtered data.

    See Also
    --------
    savgol_coeffs

    Notes
    -----
    Details on the `mode` options:

        'mirror':
            Repeats the values at the edges in reverse order. The value
            closest to the edge is not included.
        'nearest':
            The extension contains the nearest input value.
        'constant':
            The extension contains the value given by the `cval` argument.
        'wrap':
            The extension contains the values from the other end of the array.

    For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8], and
    `window_length` is 7, the following shows the extended data for
    the various `mode` options (assuming `cval` is 0)::

        mode       |   Ext   |         Input          |   Ext
        -----------+---------+------------------------+---------
        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
        'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import savgol_filter
    >>> np.set_printoptions(precision=2)  # For compact display.
    >>> x = np.array([2, 2, 5, 2, 1, 0, 1, 4, 9])

    Filter with a window length of 5 and a degree 2 polynomial.  Use
    the defaults for all other parameters.

    >>> savgol_filter(x, 5, 2)
    array([1.66, 3.17, 3.54, 2.86, 0.66, 0.17, 1.  , 4.  , 9.  ])

    Note that the last five values in x are samples of a parabola, so
    when mode='interp' (the default) is used with polyorder=2, the last
    three values are unchanged. Compare that to, for example,
    `mode='nearest'`:

    >>> savgol_filter(x, 5, 2, mode='nearest')
    array([1.74, 3.03, 3.54, 2.86, 0.66, 0.17, 1.  , 4.6 , 7.97])

    """
    if mode not in ["mirror", "constant", "nearest", "interp", "wrap"]:
        raise ValueError("mode must be 'mirror', 'constant', 'nearest' "
                         "'wrap' or 'interp'.")

    x = np.asarray(x)
    # Ensure that x is either single or double precision floating point.
    if x.dtype != np.float64 and x.dtype != np.float32:
        x = x.astype(np.float64)

    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

    if mode == "interp":
        if window_length > x.shape[axis]:
            raise ValueError("If mode is 'interp', window_length must be less "
                             "than or equal to the size of x.")

        # Do not pad. Instead, for the elements within `window_length // 2`
        # of the ends of the sequence, use the polynomial that is fitted to
        # the last `window_length` elements.
        y = convolve1d(x, coeffs, axis=axis, mode="constant")
        _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y)
    else:
        # Any mode other than 'interp' is passed on to ndimage.convolve1d.
        y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=cval)

    return y
```
请使用Python的scipy.signal库中的savgol_filter函数，完成Matlab软件sgolay函数的示例，完成MATLAB函数示例到Python程序示例的对应。

### LLM 生成 ： Python savgol_filter 函数示例

以下是使用Python中的scipy.signal库中的savgol_filter函数来完成MATLAB软件sgolay函数示例的对应Python程序示例：

#### Python
```
import numpy as np
from scipy.signal import savgol_filter

dt = 0.25
t = np.arange(0, 20, dt)

x = 5*np.sin(2*np.pi*0.2*t) + 0.5*np.random.randn(len(t))
b, g = sgolay(5, 25)

dx = np.zeros((len(x), 4))
for p in range(4):
    dx[:, p] = np.convolve(x, np.math.factorial(p)/(-dt)**p * g[:, p], mode='same')
    
import matplotlib.pyplot as plt
plt.plot(t, x, '.-', label='x')
plt.plot(t, dx[:, 0], label="x' (smoothed)")
plt.plot(t, dx[:, 1], label="x''")
plt.plot(t, dx[:, 2], label="x'''")
plt.plot(t, dx[:, 3], label="x''''")
plt.legend()
plt.title('Savitzky-Golay Derivative Estimates')
plt.show()
```
这段Python代码模拟了MATLAB软件sgolay函数的示例，并使用了Python中的scipy.signal库中的savgol_filter函数来实现相同的功能。