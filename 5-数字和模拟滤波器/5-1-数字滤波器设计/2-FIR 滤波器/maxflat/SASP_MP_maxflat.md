# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波器设计

## MATLAB函数描述：maxflat

函数来源：[MATLAB maxflat](https://ww2.mathworks.cn/help/signal/ref/maxflat.html)

### 语法

[b,a] = maxflat(n,m,Wn)
b = maxflat(n,'sym',Wn)
[b,a,b1,b2] = maxflat(n,m,Wn)
[b,a,b1,b2,sos,g] = maxflat(n,m,Wn)
[...] = maxflat(n,m,Wn,'design_flag')

### 说明

[b,a] = maxflat(n,m,Wn) 返回具有归一化截止频率 Wn 的低通巴特沃斯滤波器的第 n 阶分子系数 b 和第 m 阶分母系数 a。

b = maxflat(n,'sym',Wn) 返回对称 FIR 巴特沃斯滤波器的系数 b。n 必须是偶数。

[b,a,b1,b2] = maxflat(n,m,Wn) 返回两个多项式 b1 和 b2，它们的乘积等于分子多项式 b（即 b = conv(b1,b2)）。

[b,a,b1,b2,sos,g] = maxflat(n,m,Wn) 返回滤波器的二阶部分表示作为滤波器矩阵 sos 和增益 g。

[...] = maxflat(n,m,Wn,'designflag') 使用 designflag 指定将滤波器设计显示为表格、图形或两者。您可以使用前面语法中的任何输出组合。

### 输入参数

n — 分子系数阶
实、正标量
分子系数阶，指定为实正的标量
数据类型: single | double

m — 分母系数阶
实、正标量
分子系数阶，指定为实正的标量
数据类型: single | double

Wn —  归一化截止频率
[0,1]范围内的标量
滤波器的幅值响应等于1/√2的归一化截止频率，表示为[0,1]范围内的标量，其中1对应奈奎斯特频率。
数据类型: single | double

designflag — 滤波器设计展示
'trace' | 'plots' | 'both'
过滤器设计显示，指定为以下值之一:
'trace'用于在设计中使用的设计表的文本显示
“plots”表示滤波器幅度、群延迟、零点和极点的图
'both'表示文本显示和画图


### 输出参量

b — 分子系数
向量
分子系数，以向量形式返回。

a — 分母系数
向量
分母系数，以向量形式返回。

b1,b2 -多项式
向量
作为向量返回的多项式。b1和b2的乘积等于分子多项式b, b1包含z = -1处所有的0,b2包含其他所有的0。

sos — 二阶截面系数
矩阵
二阶截面系数，以矩阵形式返回。

g — 增益
实值标量
滤波器的增益，作为实值标量返回。
## Python函数描述：butter

函数来源：[scipy.signal.butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)

### 语法

scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)

巴特沃斯数字和模拟滤波器设计。
设计一个n阶数字(或模拟)巴特沃斯滤波器并返回滤波器系数。

### 参数

- `N`: int
- 对于“带通”和“带阻”滤波器，最终二阶部分（'sos'）矩阵的阶数是2*N，其中N是所需系统的双二次（biquad）部分的数量。

- `Wn`: array_like
- 临界频率。对于低通和高通滤波器，Wn是一个标量;对于带通和带阻滤波器，Wn是长度为2的序列。

对于巴特沃斯滤波器，这是增益下降到通带的1/√(2)的点(“-3 dB点”)。

对于数字滤波器，如果不指定fs，则将Wn单位从0归一化为1，其中1为奈奎斯特频率(因此Wn为半周期/采样，定义为2*临界频率/ fs)。如果指定了fs，则Wn的单位与fs相同。

- `btype`: {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, 可选择的
- 滤波器的类型。默认为“低通”
  
- `analog`: bool, 可选择的
- 当为True时，返回模拟滤波器，否则返回数字滤波器。
   
- `output`: {‘ba’, ‘zpk’, ‘sos’}, 可选择的
- 输出类型:分子/分母('ba')，极零('zpk')或二阶部分('sos')。默认为'ba'用于向后兼容，但'sos'应用于通用滤波。
   
- `fs`: float, 可选择的
- 数字系统的采样频率。
  
### 返回值

- `b, a`: ndarray, ndarray
- IIR滤波器的分子(b)和分母(a)多项式。仅当output='ba'时返回。
- `z, p, k`: ndarray, ndarray, float
- 零点、极点和IIR滤波器传递函数的系统增益。仅当output='zpk'时返回。
- `sos`: ndarray
- IIR滤波器的二阶截面表示。仅当output='sos'时返回。

### 注意事项
- 巴特沃斯滤波器在通带内具有最大的平坦频率响应。
  
- 'sos'输出参数在0.16.0中添加。
  
- 如果要求传递函数形式[b, a]，则会出现数值问题，因为根与多项式系数之间的转换是数值敏感的操作，即使N >= 4也是如此。建议使用SOS表示。

### 警告

- 以TF形式设计高阶和窄带IIR滤波器可能会由于浮点数值精度问题而导致滤波不稳定或不正确。考虑检查输出滤波器特性频率'freqz'，或通过output='sos'设计具有二阶部分的滤波器。

## Prompt 1 ： 应用 Python butter 函数完成 Matlab 函数示例到 Python 的对应

参考下面MATLAB代码maxflat函数例子：
```
n = 10;
m = 2;
Wn = 0.2;

[b,a] = maxflat(n,m,Wn);
fvtool(b,a)
```

和Python中的函数butter源码：
```
def butter(N, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Butterworth digital and analog filter design.

    Design an Nth-order digital or analog Butterworth filter and return
    the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter. For 'bandpass' and 'bandstop' filters,
        the resulting order of the final second-order sections ('sos')
        matrix is ``2*N``, with `N` the number of biquad sections
        of the desired system.
    Wn : array_like
        The critical frequency or frequencies. For lowpass and highpass
        filters, Wn is a scalar; for bandpass and bandstop filters,
        Wn is a length-2 sequence.

        For a Butterworth filter, this is the point at which the gain
        drops to 1/sqrt(2) that of the passband (the "-3 dB point").

        For digital filters, if `fs` is not specified, `Wn` units are
        normalized from 0 to 1, where 1 is the Nyquist frequency (`Wn` is
        thus in half cycles / sample and defined as 2*critical frequencies
        / `fs`). If `fs` is specified, `Wn` is in the same units as `fs`.

        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    buttord, buttap

    Notes
    -----
    The Butterworth filter has maximally flat frequency response in the
    passband.

    The ``'sos'`` output parameter was added in 0.16.0.

    If the transfer function form ``[b, a]`` is requested, numerical
    problems can occur since the conversion between roots and
    the polynomial coefficients is a numerically sensitive operation,
    even for N >= 4. It is recommended to work with the SOS
    representation.

    .. warning::
        Designing high-order and narrowband IIR filters in TF form can
        result in unstable or incorrect filtering due to floating point
        numerical precision issues. Consider inspecting output filter
        characteristics `freqz` or designing the filters with second-order
        sections via ``output='sos'``.

    Examples
    --------
    Design an analog filter and plot its frequency response, showing the
    critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> b, a = signal.butter(4, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Butterworth filter frequency response')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.show()

    Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz

    >>> t = np.linspace(0, 1, 1000, False)  # 1 second
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    >>> ax1.plot(t, sig)
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    >>> ax1.axis([0, 1, -2, 2])

    Design a digital high-pass filter at 15 Hz to remove the 10 Hz tone, and
    apply it to the signal. (It's recommended to use second-order sections
    format when filtering, to avoid numerical error with transfer function
    (``ba``) format):

    >>> sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
    >>> filtered = signal.sosfilt(sos, sig)
    >>> ax2.plot(t, filtered)
    >>> ax2.set_title('After 15 Hz high-pass filter')
    >>> ax2.axis([0, 1, -2, 2])
    >>> ax2.set_xlabel('Time [seconds]')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    return iirfilter(N, Wn, btype=btype, analog=analog,
                     output=output, ftype='butter', fs=fs)
```
请使用Python的butter函数，完成Matlab软件maxflat函数的示例，完成MATLAB函数示例到Python程序示例的对应。

### LLM 生成 ： Python butter 函数示例

#### Python
	import numpy as np  

	import matplotlib.pyplot as plt  

	from scipy.signal import butter, freqz  

	  

	# 滤波器阶数，注意对于最大平坦的巴特沃斯滤波器，N应为偶数  

	N = 10  

	  

	# 滤波器类型：'low', 'high', 'band', 'stop'  

	btype = 'low'  # 对应MATLAB中m=1的情况  

	  

	# 归一化截止频率（对于数字滤波器）  

	# Wn是相对于采样频率fs的一半的比例  

	# 假设采样频率为1（归一化频率），则直接使用Wn的值  

	Wn = 0.2  

	  

	# 使用butter函数设计巴特沃斯滤波器  

	b, a = butter(N, Wn, btype=btype)  

	  

	# 计算频率响应  

	w, h = freqz(b, a)  

	  

	# 绘制频率响应图  

	plt.figure()  

	plt.plot(w/np.pi, np.abs(h), 'b')     # 幅度响应  

	plt.plot(w/np.pi, 20*np.log10(np.abs(h)), 'r') # 对数幅度响应  

	plt.ylim(-150, 5)  

	plt.grid(True)  

	plt.xlabel('Normalized frequency (xπrad/sample)')  

	plt.ylabel('Magnitude')  

	plt.title('Frequency response of the Butterworth filter')  

	plt.show()

在Python代码中，我们首先导入了必要的库，然后定义了滤波器的阶数N、类型btype和归一化截止频率Wn。我们使用butter函数设计了巴特沃斯滤波器，并得到了滤波器的分子系数b和分母系数a。然后，我们使用freqz函数计算了滤波器的频率响应，并绘制了幅度响应和对数幅度响应图。

需要注意的是，MATLAB的maxflat函数和Python的butter函数在参数和默认行为上略有不同，因此在实际应用中可能需要根据具体情况调整参数以得到期望的滤波器特性。此外，由于MATLAB和Python在绘图方面有所不同，因此绘图代码也会有所差异。在Python中，我们使用matplotlib库来绘制图形。