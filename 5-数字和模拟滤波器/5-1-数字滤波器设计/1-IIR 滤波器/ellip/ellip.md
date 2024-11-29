# 信号处理仿真与应用 - 测量和特征提取 - 数字滤波器设计

## MATLAB函数描述：ellip

函数来源：[MATLAB ellip](https://ww2.mathworks.cn/help/signal/ref/ellip.html)

### 语法

[b,a] = ellip(n,Rp,Rs,Wp)
[b,a] = ellip(n,Rp,Rs,Wp,ftype)
[z,p,k] = ellip(___)
[A,B,C,D] = ellip(___)
[___] = ellip(___,'s')

### 说明


[b,a] = ellip(n,Rp,Rs,Wp) 返回一个n阶低通数字椭圆滤波器的传递函数系数，该滤波器的归一化通带边缘频率为Wp。设计的滤波器具有Rp分贝的通带峰值到峰值纹波和Rs分贝的阻带衰减，相对于通带峰值。

[b,a] = ellip(n,Rp,Rs,Wp,ftype) 设计一个低通、高通、带通或带阻椭圆滤波器，具体取决于ftype的值和Wp的元素数量。得到的带通和带阻设计是2n阶的。

[z,p,k] = ellip(___) 设计一个低通、高通、带通或带阻数字椭圆滤波器，并返回其零点、极点和增益。这个语法可以包括前述语法中的任何输入参数。

[A,B,C,D] = ellip(___) 设计一个低通、高通、带通或带阻数字椭圆滤波器，并返回指定其状态空间表示的矩阵。

[___] = ellip(___,'s') 设计一个低通、高通、带通或带阻模拟椭圆滤波器，其通带边缘角频率为Wp，通带有Rp分贝的纹波和Rs分贝的阻带衰减。


### 输入参数

n — 滤波器的阶数，即传递函数中多项式的次数。

Rp — 通带内的最大纹波（以分贝为单位）。

Rs — 阻带内的最小衰减（以分贝为单位）。

Wp — 通带边缘频率。对于低通和高通滤波器，这是一个标量；对于带通和带阻滤波器，这是一个包含两个元素的向量，分别代表通带的上边缘和下边缘频率。

ftype — 滤波器类型，可以是 ‘low’（低通）、‘high’（高通）、‘bandpass’（带通）或 ‘bandstop’（带阻）。

___ — 可以是上述函数的任何输入参数组合。

's' — 指定设计模拟滤波器。如果没有 's'，则默认设计数字滤波器。

### 输出参量

b — 传递函数的分子系数向量。

a — 传递函数的分母系数向量。

z — 滤波器的零点。

p — 滤波器的极点。

k — 滤波器的增益。

A, B, C, D — 状态空间表示的矩阵，其中 A 是系统矩阵，B 是输入矩阵，C 是输出矩阵，D 是直接传递矩阵。

___ — 根据输入参数，输出参数将与上述函数相同，但是用于模拟滤波器设计。



## Python函数描述：ellip

函数来源：[scipy.signal.ellip](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellip.html)

### 函数定义

scipy.signal.ellip(N, rp, rs, Wn, btype='low', analog=False, output='ba', fs=None)[source]

椭圆数字和模拟滤波器设计。
设计一个N阶数字或模拟椭圆滤波器，并返回滤波器系数。


### 参数

- `N`
  int
  滤波器的阶数。
- `rp`
  float
  通带内允许的最大纹波。以分贝为单位指定，为正数。
- `rs`
  float
  阻带内所需的最小衰减。以分贝为单位指定，为正数。
- `Wn`
  array_like
  一个标量或长度为2的序列，给出临界频率。对于椭圆滤波器，这是增益首次下降到-rp以下的过渡带中的点。
  对于数字滤波器，Wn的单位与fs相同。默认情况下，fs是2个半周期/样本，因此这些频率是从0到1的标准化的，其中1是奈奎斯特频率。（因此，Wn的单位是半周期/样本。）
  对于模拟滤波器，Wn是一个角频率（例如，弧度/秒）。
- `btype`
  {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, 可选
  滤波器的类型。默认是‘lowpass’。
- `analog`
  bool, 可选
  当为True时，返回模拟滤波器，否则返回数字滤波器。
- `output`
  {‘ba’, ‘zpk’, ‘sos’}, 可选
  输出类型：分子/分母（‘ba’）、零点-极点（‘zpk’）或二阶节（‘sos’）。默认是‘ba’以保持向后兼容性，但对于通用滤波应使用‘sos’。
- `fs`
  float, 可选
  数字系统的采样频率。

### 返回值

- `b, a`
  ndarray, ndarray
  IIR滤波器的分子（b）和分母（a）多项式。仅当output='ba'时返回。
- `z, p, k`
  ndarray, ndarray, float
  IIR滤波器传递函数的零点、极点和系统增益。仅当output='zpk'时返回。
- `sos`
  ndarray
  IIR滤波器的二阶节表示。仅当output='sos'时返回。



## Prompt 1 ： Lowpass Elliptic Transfer Function
参考下面Matlab代码，给出Python实现代码，并绘制信号的频率和相位：
**Matlab代码**
```
fc = 300;
fs = 1000;

[b,a] = ellip(6,10,50,fc/(fs/2));

freqz(b,a,[],fs)

subplot(2,1,1)
ylim([-100 20])
```

### LLM 生成 ：
```
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

# 定义参数
fs = 1000  # 采样频率 Hz
fc = 300    # 截止频率 Hz
rp = 10     # 通带纹波 dB
rs = 50      # 阻带纹波 dB

# 设计椭圆滤波器
b, a = signal.ellip(6, rp, rs, fc / (fs / 2), btype='low', analog=False)

# 计算频率响应
w, h = signal.freqz(b, a, worN=512,fs=1000)

# 绘制频率响应
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Lowpass Elliptic filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.show()

# 绘制相位响应
plt.figure(2)
plt.plot(w, np.degrees(np.unwrap(np.angle(h))), 'b')
plt.title('Phase Response of Lowpass Elliptic filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [degrees]')
plt.grid()
plt.show()
```


## Prompt 2 ： Bandstop Elliptic Filter
参考下面Matlab代码，给出Python实现代码，并绘制信号的频率和相位：
**Matlab代码**
```
[b,a] = ellip(3,5,50,[0.2 0.6],'stop');
freqz(b,a)
```

### LLM 生成 ：
```
import scipy.signal as signal
import numpy as np

# 定义参数
rp = 5    # 通带纹波 dB
rs = 50   # 阻带纹波 dB

# 设计椭圆滤波器
b, a = signal.ellip(3, rp, rs, [0.2,0.6], btype='stop', analog=False)

# 计算频率响应
w, h = signal.freqz(b, a, worN=512)

# 绘制频率响应
plt.figure()
plt.plot(w / np.pi, 20 * np.log10(abs(h)))
plt.title('Bandstop Elliptic Filter')
plt.xlabel('Normalized Frequency [*pi rad/sample]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.show()

# 绘制相位响应
plt.figure()
plt.plot(w / np.pi, np.degrees(np.unwrap(np.angle(h))), 'b')
plt.title('Phase Response of Bandstop Elliptic filter')
plt.xlabel('Normalized Frequency [*pi rad/sample]')
plt.ylabel('Phase [degrees]')
plt.grid()
plt.show()
```


## Prompt 3 ： Highpass Elliptic Filter
参考下面Matlab代码，给出Python实现代码，并绘制信号的频率和相位：
**Matlab代码**
```
[z,p,k] = ellip(6,3,50,300/500,"high");
sos = zp2sos(z,p,k);
freqz(sos)
```

### LLM 生成 ：
```
import numpy as np
from scipy.signal import ellip, zpk2sos
import matplotlib.pyplot as plt

# 定义参数
rp = 3     # 通带纹波
rs = 50    # 阻带衰减

# 设计椭圆滤波器
b, a = ellip(6, rp, rs, 300 / 500, btype='highpass')

# 计算频率响应
w, h = signal.freqz(b, a, worN=512)

# 绘制频率响应
plt.figure()
plt.plot(w / np.pi, 20 * np.log10(abs(h)))
plt.title(' Highpass Elliptic Filter')
plt.xlabel('Normalized Frequency [*pi rad/sample]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.show()

# 绘制相位响应
plt.figure()
plt.plot(w / np.pi, np.degrees(np.unwrap(np.angle(h))))
plt.title('Phase Response of Highpass Elliptic filter')
plt.xlabel('Normalized Frequency [*pi rad/sample]')
plt.ylabel('Phase [degrees]')
plt.grid()
plt.show()
```


## Prompt 4 ： Bandpass Elliptic Filter
参考下面Matlab代码，给出Python实现代码，并绘制信号的频率和相位：
**Matlab代码**
```
fs = 1500;

[A,B,C,D] = ellip(10,3,40,[500 560]/(fs/2));
sos = ss2sos(A,B,C,D);
freqz(sos,[],fs)
```

### LLM 生成 ：
```
import numpy as np
from scipy.signal import ellip, zpk2sos
import matplotlib.pyplot as plt

# 定义参数
fs = 1500
rp = 3     # 通带纹波
rs = 40    # 阻带衰减

# 设计椭圆滤波器
b, a = ellip(10, rp, rs, np.array([500, 560])/(fs/2), btype='pass')

# 计算频率响应
w, h = signal.freqz(b, a, worN=512,fs = 1500)

# 绘制频率响应
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Bandpass Elliptic Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.show()

# 绘制相位响应
plt.figure()
plt.plot(w, np.degrees(np.unwrap(np.angle(h))))
plt.title('Phase Response of Bandpass Elliptic filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [degrees]')
plt.grid()
plt.show()
```


## Prompt 5 ： Comparison of Analog IIR Lowpass Filters
参考下面Matlab代码，给出Python实现代码，并绘制信号的频率和相位：
**Matlab代码**
```
n = 5;
fc = 2e9;

[zb,pb,kb] = butter(n,2*pi*fc,"s");
[bb,ab] = zp2tf(zb,pb,kb);
[hb,wb] = freqs(bb,ab,4096);

[z1,p1,k1] = cheby1(n,3,2*pi*fc,"s");
[b1,a1] = zp2tf(z1,p1,k1);
[h1,w1] = freqs(b1,a1,4096);

[z2,p2,k2] = cheby2(n,30,2*pi*fc,"s");
[b2,a2] = zp2tf(z2,p2,k2);
[h2,w2] = freqs(b2,a2,4096);

[ze,pe,ke] = ellip(n,3,30,2*pi*fc,"s");
[be,ae] = zp2tf(ze,pe,ke);
[he,we] = freqs(be,ae,4096);

[zf,pf,kf] = besself(n,2*pi*fc);
[bf,af] = zp2tf(zf,pf,kf);
[hf,wf] = freqs(bf,af,4096);

plot([wb w1 w2 we wf]/(2e9*pi), ...
    mag2db(abs([hb h1 h2 he hf])))
axis([0 5 -45 5])
grid
xlabel("Frequency (GHz)")
ylabel("Attenuation (dB)")
legend(["butter" "cheby1" "cheby2" "ellip" "besself"])
```

### LLM 生成 ：
```
import numpy as np
from scipy.signal import ellip, zpk2sos
import matplotlib.pyplot as plt

# 定义参数
n = 5
fc = 2e9

b1, a1 = signal.butter(n, 2*np.pi*fc, 'low', analog=True)
w1, h1 = signal.freqs(b1, a1, worN = 4096)

b2, a2 = signal.cheby1(n, 3, 2*np.pi*fc, 'low', analog=True)
w2, h2 = signal.freqs(b2, a2, worN = 4096)

b3, a3 = signal.cheby2(n, 30, 2*np.pi*fc, 'low', analog=True)
w3, h3 = signal.freqs(b3, a3, worN = 4096)

b4, a4 = signal.ellip(n, 3, 30, 2*np.pi*fc, 'low', analog=True)
w4, h4 = signal.freqs(b4, a4, worN = 4096)

b5, a5 = signal.bessel(n, 2*np.pi*fc, 'low', analog=True)
w5, h5 = signal.freqs(b5, a5, worN = 4096)

# 绘制频率响应图
plt.figure()
plt.plot(w1/(2e9*np.pi), 20 * np.log10(np.abs(h1)), label='butter')
plt.plot(w2/(2e9*np.pi), 20 * np.log10(np.abs(h2)), label='cheby1')
plt.plot(w3/(2e9*np.pi), 20 * np.log10(np.abs(h3)), label='cheby2')
plt.plot(w4/(2e9*np.pi), 20 * np.log10(np.abs(h4)), label='ellip')
plt.plot(w5/(2e9*np.pi), 20 * np.log10(np.abs(h5)), label='bessel')

# 设置坐标轴范围和标签
plt.axis([0, 5, -45, 5])
plt.grid(True)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnitude (dB)')
plt.legend()

# 显示图表
plt.show()
```
