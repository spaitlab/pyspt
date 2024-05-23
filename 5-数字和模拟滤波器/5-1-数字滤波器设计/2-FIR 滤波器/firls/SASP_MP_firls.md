# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：firls 

函数来源：[MATLAB firls]([Least-squares linear-phase FIR filter design - MATLAB firls - MathWorks 中国](https://ww2.mathworks.cn/help/signal/ref/firls.html))

### 语法

```
b = firls(n,f,a)
b = firls(n,f,a,w)
b = firls(___,ftype)
```

### 说明

b = firls（n，f，a） 返回包含系数的行向量 的订单 - FIR 过滤器。频率和振幅特性 的结果滤波器与向量给出的滤波器匹配。bn+1nfa
b = firls（n，f，a，w） 用于对频率箱进行加权。
b = firls(___,ftype)设计反对称（奇数）滤波器，其中指定滤波器 作为微分器或希尔伯特变压器。您可以与以下任何 前面的输入语法。

### 输入参数

n— 滤波顺序
实正标量
滤波器顺序，指定为实数正标量。

f— 归一化频率点
实值向量
归一化频率点，指定为实值向量。参数必须是 在 [0， 1] 范围内，其中 1 对应于奈奎斯特频率。数量向量中的元素始终是 2 的倍数。频率必须在不递减的顺序。

fl — 希尔伯特滤波器长度
正整数标量
希尔伯特滤波器长度，指定为正整数标量。该滤波器是通过使用长度为 fl 且形状参数为 β = 8 的凯塞窗对理想的矩形滤波器加窗来创建的。
数据类型: single | double

a— 所需振幅
矢量
在 f 中指定的点处的所需振幅，指定为 向量。 并且必须具有相同的长度。这 长度必须为偶数。fa
点对之间频率下的所需振幅 （f（k）， f（k+1）） 对于 k 奇数是 连接点 （f（k）、a（k）） 和 （f（k+1），a（k+1））。
点对之间频率下的所需振幅 （f（k）， f（k+1）） 对于 k 偶数为 未指定。这些点之间的区域是过渡区域或区域 对于特定应用程序并不重要。

w— 权重
实值向量
用于调整每个频带拟合的权重，指定为实值 向量。的长度是 f 和 a 长度的一半，因此每个波段正好有一个权重。

ftype— 过滤器类型
'hilbert' |“差异化因素”
奇对称线性相位滤波器的滤波器类型（III型和IV型）， 指定为或：'hilbert''differentiator'
'hilbert'— 输出系数服从关系式 b（k） = –b（n + 2 – k）， k = 1， ...， n + 1. 这类滤波器包括希尔伯特变压器，它有一个 整个频段的振幅为 1。b
'differentiator'— 对于非零振幅带，滤波器 将误差的系数加重1楼2因此，低频的误差比高频的误差小得多 频率。对于具有振幅特性的 FIR 微分器 这些滤波器与频率成正比，可将最大相对误差降至最低（ 误差与所需振幅之比的最大值）。

### 输出参量

b— 滤波器系数
行向量
滤波器系数，以长度为 n + 1 的行向量返回。 系数按递增顺序排列。

## Python函数描述：firls

函数来源：python自带

## Prompt 1 ： 生成 Python firls 函数对应示例1

参考下面MATLAB代码的firls函数示例
```
b = firls(255,[0 0.25 0.3 1],[1 1 0 0]);
freqz(b,1)
```

请完成Python语言firls函数的编写。

### LLM 生成 ： Python firls 函数示例1

    from scipy.signal import firls, freqz
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 设计滤波器
    b = firls(257, [0, 0.25, 0.3, 1], [1, 1, 0, 0])
    
    # 计算频率响应
    w, h = freqz(b, 1, worN=8000)
    
    # 绘制频率响应
    fig, axs = plt.subplots(2, 1, figsize=(10, 7))
    
    # 幅度响应
    axs[0].plot(w / np.pi, 20 * np.log10(abs(h)))
    axs[0].set_title('Magnitude Response')
    axs[0].set_ylabel('Magnitude (dB)')
    axs[0].grid()
    
    # 相位响应
    axs[1].plot(w / np.pi, np.unwrap(np.angle(h)) * (180 / np.pi))
    axs[1].set_title('Phase Response')
    axs[1].set_ylabel('Phase (degrees)')
    axs[1].set_xlabel('Normalized Frequency (x π rad/sample)')
    axs[1].grid()
    
    plt.tight_layout()
    plt.show()

## Prompt 2 ： 生成 Python firls 函数对应示例2

参考下面MATLAB代码的firls函数示例

```
b = firls(30,[0 0.9],[0 0.9*pi],'differentiator');

zerophase(b,1)
```

请完成Python语言firls函数的编写。

### LLM 生成 ： Python firls 函数示例2

```
from scipy.signal import firls, freqz
import matplotlib.pyplot as plt
import numpy as np

# 设计微分器
# 滤波器的系数数量设为31，因此在MATLAB中对应的滤波器阶数为30
b = firls(31, [0, 0.9, 0.9, 1], [0, 0.9*np.pi, 0.9*np.pi, 0])

# 计算频率响应
w, h = freqz(b, worN=8000)
amplitude = np.abs(h)  # 零相位响应的振幅

# 绘制零相位响应
plt.plot(w / np.pi, amplitude)  # 用π标准化频率
plt.title('Zero-Phase Response')
plt.xlabel('Normalized Frequency (x π rad/sample)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```

## Prompt 3 ： 生成 Python firls 函数对应示例3

参考下面MATLAB代码的firls函数示例

```
F = [0 0.3 0.4 0.6 0.7 0.9]; 
A = [0 1.0 0.0 0.0 0.5 0.5];
b = firls(24,F,A,'hilbert');
[H,f] = freqz(b,1,512,2);
plot(f,abs(H))
hold on
for i = 1:2:6, 
   plot([F(i) F(i+1)],[A(i) A(i+1)],'r--')
end
legend('firls design','Ideal')
grid on
xlabel('Normalized Frequency (\times\pi rad/sample)')
ylabel('Magnitude')
```

请完成Python语言firls函数的编写。

### LLM 生成 ： Python firls 函数示例3

```
from scipy.signal import firls, freqz
import matplotlib.pyplot as plt
import numpy as np

# 设定频率和振幅，确保设计是反对称的
F = [0, 0.3, 0.4, 0.6, 0.7, 0.9, 0.9, 1] 
A = [0, 1.0, 0.0, 0.0, 0.5, 0.5, 0, 0]
# 对于希尔伯特变换器，我们需要滤波器系数的数量是奇数，因此阶数（numtaps-1）应该是偶数
numtaps = 25

# 频率响应的权重，提高0.9之后的权重以实现更陡峭的截止
weights = [1] * 3 + [100]

b = firls(numtaps, F, A, weight=weights, fs=2)

# 计算并绘制频率响应
w, h = freqz(b, worN=512)
f = w / np.pi
plt.plot(f, np.abs(h))

# 绘制理想的频率响应
for i in range(0, len(F), 2):
    plt.plot(F[i:i+2], A[i:i+2], 'r--')

plt.legend(['firls design', 'Ideal'])
plt.grid(True)
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Magnitude')
plt.show()
```

## Prompt 4 ： 生成 Python firls 函数对应示例4

参考下面MATLAB代码的firls函数示例

```
bhi = firls(18,[0 0.45 0.55 1],[1 1 0 0],[1 100]);
blo = firls(18,[0 0.45 0.55 1],[1 1 0 0],[100 1]);
b = firls(18,[0 0.45 0.55 1],[1 1 0 0],[1 1]);
zerophase(bhi,1)
hold on
zerophase(blo,1)
zerophase(b,1)
hold off
ylim([-.2 1.2])
legend(["bhi: w = [1 100]" "blo: w = [100 1]" "b: w = [1 1]"])
```

请完成Python语言firls函数的编写。

### LLM 生成 ： Python firls 函数示例3

```
from scipy.signal import firls, freqz
import matplotlib.pyplot as plt
import numpy as np

# 设定频率点和对应的振幅值
freq_points = [0, 0.45, 0.55, 1]
desired_amplitude = [1, 1, 0, 0]

# 设计三个滤波器，使用不同的权重
weights_bhi = [1, 100]  # 停带权重高
bhi = firls(19, freq_points, desired_amplitude, weights_bhi, fs=2)

weights_blo = [100, 1]  # 通带权重高
blo = firls(19, freq_points, desired_amplitude, weights_blo, fs=2)

weights_b = [1, 1]  # 权重相等
b = firls(19, freq_points, desired_amplitude, weights_b, fs=2)

# 计算并绘制三个滤波器的零相位响应
w, h_bhi = freqz(bhi, worN=512)
w, h_blo = freqz(blo, worN=512)
w, h_b = freqz(b, worN=512)

# 绘制零相位响应
plt.plot(w/np.pi, abs(h_bhi), label="bhi: w = [1 100]")
plt.plot(w/np.pi, abs(h_blo), label="blo: w = [100 1]")
plt.plot(w/np.pi, abs(h_b), label="b: w = [1 1]")

plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Magnitude')
plt.legend()
plt.ylim([-.2, 1.2])
plt.grid(True)
plt.show()
```