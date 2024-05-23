# 信号处理仿真与应用 - 测量和特征提取 - 数字滤波器设计

## MATLAB函数描述：ellipord

函数来源：[MATLAB ellipord](https://ww2.mathworks.cn/help/signal/ref/ellipord.html)

### 语法

[n,Wn] = ellipord(Wp,Ws,Rp,Rs)
[n,Wn] = ellipord(Wp,Ws,Rp,Rs,'s')

### 说明

[n,Wn] = ellipord(Wp,Ws,Rp,Rs) 返回数字椭圆滤波器的最低阶数，n，该滤波器在通带内具有不超过Rp分贝的纹波，在阻带内至少有Rs分贝的衰减。Wp和Ws分别是滤波器的通带和阻带边缘频率，从0到1标准化，其中1对应于π rad/sample。相应的截止频率Wn的标量（或向量）也会被返回。要设计一个椭圆滤波器，请将输出参数n和Wn作为输入传递给ellip函数。

[n,Wn] = ellipord(Wp,Ws,Rp,Rs,'s') 为模拟椭圆滤波器找到最小的阶数n和截止频率Wn。在指定频率Wp和Ws时，使用每秒弧度。通带或阻带可以是无限的。


### 输入参数

Wp — 通带边缘频率。对于数字滤波器，这是一个标量或一个包含两个元素的向量，代表通带的下边缘和上边缘频率，通常是归一化的频率，范围从0到1，其中1对应于奈奎斯特频率。对于模拟滤波器，Wp是角频率（例如，以rad/s为单位）的标量或向量。

Ws — 阻带边缘频率。与Wp类似，Ws也是一个标量或向量，代表阻带的下边缘和上边缘频率，归一化或以角频率表示。

Rp — 通带内的最大纹波，以分贝（dB）为单位。这是一个正数，表示通带内信号增益的最大波动。

Rs — 阻带内的最小衰减，以分贝（dB）为单位。这也是一个正数，表示阻带内信号增益的最小衰减。

's' — 可选字符串参数，仅在MATLAB中。当包含此参数时，函数为模拟滤波器计算参数。在SciPy中，使用analog参数来指定模拟滤波器设计。

### 输出参量

n — 计算出的滤波器的最小阶数。这是一个整数，表示滤波器的复杂性或传递函数中多项式的次数。

Wn — 对应的截止频率。对于数字滤波器，Wn是归一化的截止频率的标量或向量。对于模拟滤波器，Wn是角频率的标量或向量。


## Python函数描述：ellipord

函数来源：[scipy.signal.ellipord](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellipord.html)

### 函数定义

scipy.signal.ellipord(wp, ws, gpass, gstop, analog=False, fs=None)[source]
椭圆（Cauer）滤波器阶数选择。
返回一个最低阶数的数字或模拟椭圆滤波器，该滤波器在通带内损失不超过gpass分贝，并且在阻带内至少有gstop分贝的衰减。


### 参数

- `wp, ws`
  float
  通带和阻带边缘频率。
  对于数字滤波器，这些频率的单位与fs相同。默认情况下，fs是2个半周期/样本，因此这些频率是从0到1的标准化的，其中1是奈奎斯特频率。（因此，wp和ws的单位是半周期/样本。）例如：
  - 低通：wp = 0.2, ws = 0.3
  - 高通：wp = 0.3, ws = 0.2
  - 带通：wp = [0.2, 0.5], ws = [0.1, 0.6]
  - 带阻：wp = [0.1, 0.6], ws = [0.2, 0.5]
  对于模拟滤波器，wp和ws是角频率（例如，弧度/秒）。
- `gpass`
  float
  通带内的最大损耗（分贝）。
- `gstop`
  float
  阻带内的最小衰减（分贝）。
- `analog`
  bool, 可选
  当为True时，返回模拟滤波器，否则返回数字滤波器。
- `fs`
  float, 可选
  数字系统的采样频率。
  

### 返回值

- `ord`
  int
  满足规格的椭圆（Cauer）滤波器的最低阶数。
- `wn`
  ndarray or float
  用于与ellip一起使用以获得滤波器结果的切比雪夫自然频率（“3dB频率”）。如果指定了fs，则wn的单位与fs相同，并且必须将fs也传递给ellip。


## Prompt 1 ： Lowpass Elliptic Filter Order
参考下面Matlab代码，给出Python实现代码：
**Matlab代码**
```
Wp = 40/500;
Ws = 150/500;
Rp = 3;
Rs = 60;
[n,Wp] = ellipord(Wp,Ws,Rp,Rs)
[z,p,k] = ellip(n,Rp,Rs,Wp);
sos = zp2sos(z,p,k);
freqz(sos,512,1000)
title(sprintf('n = %d Elliptic Lowpass Filter',n))
```

### LLM 生成 ：
```
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

# 定义参数
Wp = 40/500
Ws = 150/500
Rp = 3
Rs = 60

# 计算滤波器阶数和新的归一化截止频率
n, Wp = signal.ellipord(Wp, Ws, Rp, Rs)

# 设计切比雪夫II型带通滤波器
b, a = signal.ellip(n, Rp, Rs, Wp, btype='low')

# 计算频率响应
w, h = signal.freqz(b, a, worN=512, fs=1000)

# 绘制频率响应
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.title("n = %d" % n +" Elliptic Lowpass Filter")
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.show()

# 绘制相位响应
plt.figure()
plt.plot(w, np.degrees(np.unwrap(np.angle(h))), 'b')
plt.title('Phase')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [degrees]')
plt.grid()
plt.show()
```

## Prompt 2 ： Bandpass Elliptic Filter Order
参考下面Matlab代码，给出Python实现代码：
**Matlab代码**
```
Wp = [60 200]/500;
Ws = [50 250]/500;
Rp = 3;
Rs = 40;

[n,Wp] = ellipord(Wp,Ws,Rp,Rs)
[z,p,k] = ellip(n,Rp,Rs,Wp);
sos = zp2sos(z,p,k);

freqz(sos,512,1000)
title(sprintf('n = %d Elliptic Bandpass Filter',n))
```

### LLM 生成 ：
```
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

# 定义参数
Wp = [60/500, 200/500]
Ws = [50/500, 250/500]
Rp = 3
Rs = 40

# 计算滤波器阶数和新的归一化截止频率
n, Wp = signal.ellipord(Wp, Ws, Rp, Rs)

# 设计切比雪夫II型带通滤波器
b, a = signal.ellip(n, Rp, Rs, Wp, btype='pass')

# 计算频率响应
w, h = signal.freqz(b, a, worN=512, fs=1000)

# 绘制频率响应
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.title("n = %d" % n +" Elliptic Lowpass Filter")
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.show()

# 绘制相位响应
plt.figure()
plt.plot(w, np.degrees(np.unwrap(np.angle(h))), 'b')
plt.title('Phase')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [degrees]')
plt.grid()
plt.show()
```