# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：impinvar 

函数来源：[MATLAB impinvar](https://ww2.mathworks.cn/help/signal/ref/impinvar.html?searchHighlight=impinvar&s_tid=srchtitle_support_results_1_impinvar#f7-1097542)

### 语法

[bz,az] = impinvar(b,a,fs)

[bz,az] = impinvar(b,a,fs,tol)
### 说明

[bz,az] = impinvar(b,a,fs) 创建一个数字滤波器，其分子系数和分母系数分别为 bz 和 az。该数字滤波器的脉冲响应等同于具有系数 b 和 a 的模拟滤波器的脉冲响应，并且通过 1/fs 进行缩放，其中 fs 是采样频率。

[bz,az] = impinvar(b,a,fs,tol) 使用由 tol 指定的容差来确定极点是否重复。
### 输入参数

b, a — 模拟滤波器传递函数系数
向量
模拟滤波器传递函数的系数，以向量的形式指定。
示例：[b,a] = butter(6,2*pi*10,'s') 指定了一个截止频率为 10 Hz 的 6 阶巴特沃斯滤波器。
数据类型：single | double

fs — 采样率
1 Hz（默认值） | 正标量
采样率，以正标量的形式指定。
数据类型：single | double

tol — 容差
0.001（默认值） | 正标量
容差，以正标量的形式指定。容差用于确定极点是否重复。较大的容差增加了 impinvar 将相邻位置的极点解释为重数（重复的）的可能性。默认的容差对应于极点大小的 0.1%。极点值的精度仍然受到 roots 函数可获得的精度限制。
数据类型：single | double

### 输出参量

bz, az — 数字滤波器传递函数系数
向量
返回的数字滤波器传递函数系数，以向量的形式给出。



## Python函数描述：bilinear

函数来源：scipy.signal库

### 模数滤波器转换函数定义：

def bilinear(b, a, fs):
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``2*fs*(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    b : array_like
        Numerator of the analog filter transfer function.
    a : array_like
        Denominator of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.

    Returns
    -------
    b : ndarray
        Numerator of the transformed digital filter transfer function.
    a : ndarray
        Denominator of the transformed digital filter transfer function.



这段代码定义了一个名为 `bilinear` 的函数，可以将连续时间模拟滤波器转换为离散时间数字滤波器，通过对模拟滤波器的分子（b）和分母（a）多项式进行双线性变换，得到数字滤波器的分子（bz）和分母（az）多项式。

### 参数

- `b` : 数组类型
模拟滤波器传递函数的分子。

- `a` : 数组类型
模拟滤波器传递函数的分母。

- `fs` : 浮点数
采样率，以常规频率单位给出（例如，赫兹）。此函数中不进行预变形
### 返回值
- `b` : ndarray（多维数组）
转换后的数字滤波器传递函数的分子。

- `a` : ndarray（多维数组）
转换后的数字滤波器传递函数的分母。
### 注意事项
- bilinear 函数使用双线性变换（Tustin 方法），这种变换在模拟滤波器的频域和数字滤波器之间建立映射关系。然而，这种变换可能会导致频域的预畸变（warping），特别是在处理高频信号时。
- 与 impinvar（脉冲响应不变法）相比，bilinear 更适合于带宽较窄的滤波器，而 impinvar 更适合于保持脉冲响应的形状。
### 函数工作原理
1. 通过双线性变换，将模拟滤波器的 s 平面上的极点和零点映射到数字滤波器的 z 平面上。
2. 使用 2 * fs * (z - 1) / (z + 1) 替代 s，其中 fs 是采样频率，从而实现从 s 域到 z 域的转换。
3. 转换后的极点和零点决定了数字滤波器的传递函数。
### 使用场景
bilinear 函数适用于需要快速实现和计算的数字滤波器设计，特别是在滤波器的频带较窄时。

它广泛用于数字信号处理领域，如音频处理、图像处理和通信系统中的滤波器设计。
### 改进建议
- 可以增加参数以允许用户选择不同的预畸变选项，以改善在高频区域的性能。
- 可以提供对输入模拟滤波器的极点和零点的验证，确保它们适合使用双线性变换进行转换。
- 可以增加对转换后数字滤波器性能的评估，如群延迟和频率响应的分析，以帮助用户理解转换对滤波器性能的影响。


## Prompt 1 ： 应用 Python 的 bilinear 函数做 MATLAB 示例对应

参考下面MATLAB代码
```
% 使用脉冲不变性将六阶模拟巴特沃斯低通滤波器转换为数字滤波器。指定 10 Hz 的采样率和 2 Hz 的截止频率。 显示滤波器的频率响应。
f = 2;
fs = 10;

[b,a] = butter(6,2*pi*f,'s');
[bz,az] = impinvar(b,a,fs);

freqz(bz,az,1024,fs)
```
我们采用Python语言实现将六阶模拟巴特沃斯低通滤波器转换为数字滤波器
```
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 设定截止频率和采样频率
f = 2
fs = 10

# 创建一个6阶的巴特沃斯模拟滤波器
b, a = signal.butter(6, 2 * np.pi * f, analog=True, output='ba')

# 使用双线性变换方法将模拟滤波器转换为数字滤波器
bz, az = signal.bilinear(b, a, fs)


```

### LLM 生成 ： 调用程序实现将六阶模拟巴特沃斯低通滤波器转换为数字滤波器并展示出

```
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 设定截止频率和采样频率
f = 2
fs = 10

# 创建一个6阶的巴特沃斯模拟滤波器
b, a = signal.butter(6, 2 * np.pi * f, analog=True, output='ba')

# 使用双线性变换方法将模拟滤波器转换为数字滤波器
bz, az = signal.bilinear(b, a, fs)

# 计算并绘制数字滤波器的频率响应
w, h = signal.freqz(bz, az, 1024, fs=fs)

# 绘制幅度响应
plt.figure()
plt.plot(w, 20 * np.log10(np.abs(h)))
plt.title('Digital Filter Frequency Response')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Amplitude (dB)')
plt.grid()
plt.show()

# 绘制相位响应
plt.figure()
plt.plot(w, np.unwrap(np.angle(h, deg=False)))
plt.title('Digital Filter Phase Response')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Phase (radians)')
plt.grid()
plt.show()
```
