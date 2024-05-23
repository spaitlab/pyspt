# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波器设计

## MATLAB函数描述：rcosdesign

函数来源：[MATLAB rcosdesign](https://ww2.mathworks.cn/help/signal/ref/rcosdesign.html)

### 语法

b = rcosdesign(beta,span,sps)

b = rcosdesign(beta,span,sps,shape)

### 说明

b = rcosdesign(beta,span,sps)返回系数b，该系数b对应于具有由beta指定的滚降因子的平方根提升余弦FIR滤波器。滤波器被截断为'span'符号，每个符号周期包含'sps'个样本。滤波器的阶数'sps*span'必须是偶数。滤波器能量为1。

b = rcosdesign(beta,span,sps,shape)当你将shape设置为“sqrt”时返回一个平方根凸起余弦滤波器，当你将'shape'设置为“normal”时返回一个正常凸起余弦FIR滤波器。

### 输入参数

- beta — 滚降系数
  
  实非负标量
  
  滚降系数，被指定为不大于1的实非负标量。滚降因子决定了滤波器的多余带宽。零滚降对应于矩形滤波器，单位滚降对应于纯提升余弦。
  
  数据类型: single | double

- span -符号数

    正整数标量

    符号数，指定为正整数标量。

    数据类型: single | double | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64

- sps -每个符号的样本数
  
    正整数标量

    每个符号的样本数(过采样因子)，指定为正整数标量。

    数据类型: single | double | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64

- shape -升余弦窗的形状
  
    "sqrt"(默认)| "normal"

    升余弦窗的形状，指定为“normal”或“sqrt”。


### 输出参量

- b - FIR滤波器系数

    行向量

    升余弦滤波器系数，作为行向量返回。

    数据类型: double

## Python函数描述：rcosfilter和rrcosfilter

函数来源：

[commpy.filters.rcosfilter](https://commpy.readthedocs.io/en/latest/generated/commpy.filters.rcosfilter.html)

[commpy.filters.rrcosfilter](https://commpy.readthedocs.io/en/latest/generated/commpy.filters.rrcosfilter.html)

### 语法


### 函数rcosfilter

rcosfilter(N, alpha, Ts, Fs)

产生一个上升余弦(RC)滤波器(FIR)脉冲响应。

#### 参数

- `N`: (int)  样本中滤波器的长度。

- `alpha`:  (float)   滚降因子(有效值为[0,1])

- `Ts`: (float)
    符号周期，单位为秒
  
- `Fs`: (float) 采样率(Hz)
  
#### 返回值

- `time_idx`: 1-D ndarray (float)   包含脉冲响应的时间指数(以秒为单位)的数组。
- `h_rc`: 1-D ndarray (float)   上升余弦滤波器的脉冲响应。

### 函数rrcosfilter

rrcosfilter(N, alpha, Ts, Fs)

产生一个根升余弦(RRC)滤波器(FIR)脉冲响应。

#### 参数

- `N`: int  样本中滤波器的长度。

- `alpha`:  float   滚降因子(有效值为[0,1])

- `Ts`: float   符号周期，单位为秒
  
- `Fs`: float   采样率(Hz)
  
#### 返回值

- `time_idx`: 1-D ndarray (float)   包含脉冲响应的时间指数(以秒为单位)的数组。
- `h_rc`: 1-D ndarray (float)   根升余弦滤波器的脉冲响应。

## Prompt 1 ： 应用 Python rrcosfilter 函数完成 Matlab 函数示例设计平方根升余弦滤波器

参考下面MATLAB代码rcosdesign函数例子：
```
h = rcosdesign(0.25,6,4);
mx = max(abs(h-rcosdesign(0.25,6,4,"sqrt")))
impz(h)
```

和Python中库commpy的函数:rrcosfilter(N, alpha, Ts, Fs)：
```
def rrcosfilter(N, alpha, Ts, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.

    Parameters
    ----------
    N : int
        Length of the filter in samples.

    alpha : float
        Roll off factor (Valid values are [0, 1]).

    Ts : float
        Symbol period in seconds.

    Fs : float
        Sampling Rate in Hz.

    Returns
    ---------

    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.

    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.
    """

    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
        elif alpha != 0 and t == Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif alpha != 0 and t == -Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
                    4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
                    (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)

    return time_idx, h_rrc
```
请使用Python的commpy库中的rrcosfilter函数，完成Matlab软件rcosdesign函数的示例，完成MATLAB函数示例到Python程序示例的对应。

### LLM 生成 ： Python rrcosfilter 函数示例

#### Python

	import numpy as np  

	import matplotlib.pyplot as plt  

	  

	# 定义rrcosfilter函数（如果尚未定义）  

	def rrcosfilter(N, alpha, Ts, Fs):  

	     """
        Generates a root raised cosine (RRC) filter (FIR) impulse response.

        Parameters
        ----------
        N : int
            Length of the filter in samples.

        alpha : float
            Roll off factor (Valid values are [0, 1]).

        Ts : float
            Symbol period in seconds.

        Fs : float
            Sampling Rate in Hz.

        Returns
        ---------

        time_idx : 1-D ndarray of floats
            Array containing the time indices, in seconds, for
            the impulse response.

        h_rrc : 1-D ndarray of floats
            Impulse response of the root raised cosine filter.
        """

        T_delta = 1/float(Fs)
        time_idx = ((np.arange(N)-N/2))*T_delta
        sample_num = np.arange(N)
        h_rrc = np.zeros(N, dtype=float)

        for x in sample_num:
            t = (x-N/2)*T_delta
            if t == 0.0:
                h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
            elif alpha != 0 and t == Ts/(4*alpha):
                h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                        (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
            elif alpha != 0 and t == -Ts/(4*alpha):
                h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                        (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
            else:
                h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
                        4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
                        (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)


	    return time_idx, h_rrc  

	  

	# 参数设置，模仿MATLAB的rcosdesign函数  

	beta = 0.25  # 滚降因子  

	span = 6     # 滤波器跨度（在符号时间内的倍数）  

	num_taps = 4 # 滤波器系数数量  

	  

	# 符号周期和采样率  

	Ts = 1.0  

	Fs = 1.0 / Ts  

	  

	# 计算滤波器长度  

	N = span * num_taps  

	  

	# 生成根升余弦滤波器的脉冲响应  

	time_idx, h_rrc = rrcosfilter(N, beta, Ts, Fs)  

	  

	# 因为Python函数直接返回根升余弦滤波器，不需要像MATLAB那样指定"sqrt"  

	# 所以我们直接比较h_rrc和重新计算的h_rrc_recomputed  

	h_rrc_recomputed = rrcosfilter(N, beta, Ts, Fs)[1]  

	  

	# 计算最大绝对差异  

	mx = np.max(np.abs(h_rrc - h_rrc_recomputed))  

	print(f"Maximum absolute difference between h_rrc and recomputed h_rrc: {mx}")  

	  

	# 绘制滤波器的脉冲响应  

	plt.stem(time_idx, h_rrc, use_line_collection=True)  

	plt.xlabel('Time (s)')  

	plt.ylabel('Amplitude')  

	plt.title('Root Raised Cosine Filter Impulse Response')  

	plt.grid(True)  

	plt.show()

注意，我们没有像MATLAB的rcosdesign函数那样提供一个选项来生成平方根滤波器，因为rrcosfilter函数本身就是为生成根升余弦滤波器设计的。因此，我们直接比较了两次调用rrcosfilter函数得到的脉冲响应，并计算了它们之间的最大绝对差异。

如果你需要更精确地模仿MATLAB的rcosdesign函数的行为，包括能够生成平方根滤波器，你可能需要修改rrcosfilter函数或寻找一个更完整的Python库，该库提供了与MATLAB的rcosdesign函数更相似的功能。然而，就当前提供的代码而言，上述示例已经满足了你的要求。

## Prompt 2 ： 应用 Python rrcosfilter 函数完成 Matlab 函数示例计算升余弦和平方根升余弦滤波器的脉冲响应

参考下面MATLAB代码rcosdesign函数例子：
```
rf = 0.25;
span = 4;
sps = 3;

h1 = rcosdesign(rf,span,sps,"normal");
impz(h1)

h2 = rcosdesign(rf,span,sps,"sqrt");
impz(h2)

h3 = conv(h2,h2,"same");

stem(0:span*sps,[h1/max(abs(h1));h3/max(abs(h3))]',"filled")
xlabel("Samples")
ylabel("Normalized Amplitude")
legend("h1","h2 * h2")
```

参考Python中库commpy的函数:rrcosfilter(N, alpha, Ts, Fs)：

def rrcosfilter(N, alpha, Ts, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.
    Parameters
    ----------
    N : int
        Length of the filter in samples.
    alpha : float
        Roll off factor (Valid values are [0, 1]).
    Ts : float
        Symbol period in seconds.
    Fs : float
        Sampling Rate in Hz.
    Returns
    ---------
    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.

    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.
    """
    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)
    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
        elif alpha != 0 and t == Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif alpha != 0 and t == -Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
                    4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
                    (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)
    return time_idx, h_rrc
和Python中库commpy的函数:rcosfilter(N, alpha, Ts, Fs)：
```
def rcosfilter(N, alpha, Ts, Fs):
    """
    Generates a raised cosine (RC) filter (FIR) impulse response.
    Parameters
    ----------
    N : int
        Length of the filter in samples.
    alpha : float
        Roll off factor (Valid values are [0, 1]).
    Ts : float
        Symbol period in seconds.

    Fs : float
        Sampling Rate in Hz.
    Returns
    -------
    time_idx : 1-D ndarray (float)
        Array containing the time indices, in seconds, for the impulse response.
    h_rc : 1-D ndarray (float)
        Impulse response of the raised cosine filter.
    """
    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    sample_num = np.arange(N)
    h_rc = np.zeros(N, dtype=float)
    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rc[x] = 1.0
        elif alpha != 0 and t == Ts/(2*alpha):
            h_rc[x] = (np.pi/4)*(np.sin(np.pi*t/Ts)/(np.pi*t/Ts))
        elif alpha != 0 and t == -Ts/(2*alpha):
            h_rc[x] = (np.pi/4)*(np.sin(np.pi*t/Ts)/(np.pi*t/Ts))
        else:
            h_rc[x] = (np.sin(np.pi*t/Ts)/(np.pi*t/Ts))* \
                    (np.cos(np.pi*alpha*t/Ts)/(1-(((2*alpha*t)/Ts)*((2*alpha*t)/Ts))))
    return time_idx, h_rc
```
请使用Python的commpy库中的rrcosfilter函数和rcosfilter函数，完成Matlab软件rcosdesign函数的示例，完成MATLAB函数示例到Python程序示例的对应。

### LLM 生成 ： Python rrcosfilter 和 rcosfilter 函数示例

#### Python
```
import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rcosfilter, rrcosfilter

#Parameters
rf = 0.25
span = 4
sps = 3

#Generating Root Raised Cosine (RRC) filter
N = span * sps * 2 + 1  # Filter length
alpha = rf
Ts = 1.0  # Symbol period in seconds
Fs = sps  # Sampling rate in Hz

time_idx_rrc, h_rrc = rrcosfilter(N, alpha, Ts, Fs)

#Generating Raised Cosine (RC) filter
time_idx_rc, h_rc = rcosfilter(N, alpha, Ts, Fs)

#Plotting impulse responses
plt.figure()
plt.stem(time_idx_rrc, h_rrc, use_line_collection=True)
plt.xlabel("Samples")
plt.ylabel("Normalized Amplitude")
plt.title("Root Raised Cosine (RRC) Filter")
plt.grid(True)
plt.show()

plt.figure()
plt.stem(time_idx_rc, h_rc, use_line_collection=True)
plt.xlabel("Samples")
plt.ylabel("Normalized Amplitude")
plt.title("Raised Cosine (RC) Filter")
plt.grid(True)
plt.show()
```
这段代码将生成Root Raised Cosine (RRC)和Raised Cosine (RC)滤波器的冲激响应，并绘制它们的图形，以模拟MATLAB中rcosdesign函数的示例。

## Prompt 3 ： 应用 Python rcosfilter 函数完成 Matlab 函数示例通过升余弦滤波器传递信号

参考下面MATLAB代码rcosdesign函数示例：
```
rolloff = 0.25;     % Rolloff factor
span = 6;           % Filter span in symbols
sps = 4;            % Samples per symbol
b = rcosdesign(rolloff, span, sps);
d = 2*randi([0 1], 100, 1) - 1;
x = upfirdn(d, b, sps);
r = x + randn(size(x))*0.01;
y = upfirdn(r, b, 1, sps);
请使用Python的commpy库中的rrcosfilter函数和rcosfilter函数，完成Matlab软件rcosdesign函数的示例，完成MATLAB函数示例到Python程序示例的对应。
```

### LLM 生成 ： Python rrcosfilter 和 rcosfilter 函数示例

```
import numpy as np
from commpy.filters import rcosfilter, rrcosfilter
from scipy.signal import upfirdn
import matplotlib.pyplot as plt

# Parameters
rolloff = 0.25
span = 6
sps = 4

# Generate Root Raised Cosine (RRC) filter coefficients
N_rrc = span * sps * 2 + 1
alpha = rolloff
Ts = 1.0
Fs = sps
_, h_rrc = rrcosfilter(N_rrc, alpha, Ts, Fs)

# Generate Raised Cosine (RC) filter coefficients
N_rc = span * sps * 2 + 1
_, h_rc = rcosfilter(N_rc, alpha, Ts, Fs)

# Generate random data
d = 2*np.random.randint(0, 2, 100) - 1

# Upsample and filter data using RRC filter
x = upfirdn(h_rrc, d, sps)

# Add noise
r = x + np.random.normal(0, 0.01, len(x))

# Filter received signal using RRC filter
y = upfirdn(h_rrc, r, 1, sps)

# Plot original and received signals
plt.figure()
plt.plot(x, label='Transmitted Signal')
plt.plot(y, label='Received Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Transmitted and Received Signals')
plt.legend()
plt.grid(True)
plt.show()
```

这段代码模拟了MATLAB中rcosdesign函数示例的功能，包括生成随机数据，使用RRC滤波器进行上采样和滤波，添加噪声，再次使用RRC滤波器进行滤波，并绘制了原始信号和接收信号的图形。