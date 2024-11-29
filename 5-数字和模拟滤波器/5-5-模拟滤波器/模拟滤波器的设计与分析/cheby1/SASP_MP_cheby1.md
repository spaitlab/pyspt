# 信号处理仿真与应用 - 模拟滤波器 - 模拟滤波器的设计与分析

## MATLAB函数描述：cheby1

函数来源：[MATLAB cheby1](https://ww2.mathworks.cn/help/signal/ref/cheby1.html)
### 语法
[b,a] = cheby1(n,Rp,Wp)
[b,a] = cheby1(n,Rp,Wp,ftype)
[z,p,k] = cheby1(___)
[A,B,C,D] = cheby1(___)
[___] = cheby1(___,'s')


### 说明
[b，a] = cheby1（n，Rp，Wp） 返回 th阶的传递函数系数 低通数字切比雪夫 I 型滤波器，带归一化 通带边沿频率和分贝 峰峰值通带纹波。nWpRp


[b，a] = cheby1（n，Rp，Wp，ftype） 设计 低通、高通、带通或带阻切比雪夫 I 型 filter，具体取决于 的值和 的元素数。生成的带通 带阻设计为 2 级。ftypeWpn

注意：有关影响的数值问题的信息，请参阅限制 形成传递函数。


[z,p,k] = cheby1(___)设计 低通、高通、带通或带阻数字切比雪夫 I 型滤波器，并返回其零点、极点和增益。这 语法可以包含以前语法中的任何输入参数。


[A,B,C,D] = cheby1(___)设计 低通、高通、带通或带阻数字切比雪夫 I 型滤波器，并返回指定其状态空间的矩阵 表示法。

[___] = cheby1(___,'s')设计 低通、高通、带通或带阻模拟切比雪夫 I 型滤波器，具有通带边缘角频率 Wp 和 Rp 分贝 通带纹波。

### 输入参数
n— 筛选顺序 整数标量 筛选器顺序，指定为整数标量。对于带通和带阻设计，表示滤波器阶数的一半。数据类型：double

Rp— 峰峰值通带纹波 正标量 峰峰值通带纹波，指定为正标量 以分贝表示。如果您的规格 l 采用线性单位，则可以 使用 = 40 log 将其转换为分贝Rp10（（1+l）/（1–l））。数据类型：double

Wp— 通带边缘频率 标量 |二元素向量 通带边缘频率，指定为标量或双元件 向量。通带边沿频率是 滤波器的幅度响应为 –Rp 分贝。通带纹波值越小，导致更宽的过渡带。如果是标量，则设计 具有边沿频率的低通或高通滤波器。如果 是二元素向量，其中 < ，则设计 具有较低边缘频率的带通或带阻滤波器，以及 更高的边缘频率。Wp[w1 w2]w1w2cheby1w1w2.对于数字滤波器，通带边缘频率 必须介于 0 和 1 之间，其中 1 对应于奈奎斯特速率 - 一半 采样率或π rad/样本。对于模拟滤波器，必须表示通带边缘频率 以弧度每秒为单位，可以取任何正值。数据类型：double

ftype— 过滤器类型
“低” |“带通” |“高” |“停止”
筛选器类型，指定为以下类型之一：

'low'指定低通滤波器 通带边沿频率 Wp。 是 标量 的默认值。'low'Wp

'high'指定高通滤波器 具有通带边沿频率。Wp

'bandpass'指定带通 如果 为 ，则为 2N 阶的滤波器 一个双元素向量。 是默认值 when 有两个元素。Wp'bandpass'Wp

'stop'指定带阻滤波器 如果 2 阶是双元素 向量。nWp

### 输出参量
b,a— 传递函数系数行 向量 滤波器的传递函数系数，对于低通和高通滤波器，作为长度为 n + 1 的行向量返回，对于带通和带阻滤波器，返回为长度为 2 + 1 的行向量。数据类型：double

z,p,k— 零点、极点和增益 列向量，标量 滤波器的零点、极点和增益返回为长度为 n 的两个列向量（带通和带阻设计为 2）和一个标量。数据类型：double

A,B,C,D— 状态空间矩阵矩阵 筛选器的状态空间表示，以矩阵形式返回。 如果 m = n 为 低通和高通设计，带通和带阻滤波器的 M = 2， 然后是 m × m，是 m × 1，是 1 × m，是 1 × 1。数据类型：double

## Python函数描述：cheby1
函数来源：自定义
### cheby1函数定义：

	# 定义cheby1函数
	import numpy as np
	from scipy import signal

	def cheby1_filter(order, rp, Wn, btype='low', analog=False, fs=None):
    	b, a = signal.cheby1(order, rp, Wn, btype=btype, analog=analog, fs=fs)
    	return b, a

### 参数
- `order`：滤波器的阶数。
- `rp`：在通带中的最大允许波纹幅度。
- `Wn`：归一化的截止频率，取值范围为0到1。
- `btype`：滤波器类型，这里默认为低通滤波器。
- `analog`：指定滤波器是否为模拟滤波器，默认为False，表示数字滤波器。
- `fs`：数字滤波器的采样频率。

### 返回值
函数返回计算得到的滤波器系数b和a

## Prompt 1 ： 设计一个具有 10 dB 通带纹波和通带边缘频率为 300 Hz 的 6 阶低通切比雪夫 I 型滤波器，对于以 1000 Hz 采样的数据，对应于0.6πrad/sample。绘制其幅度和相位响应。使用它来过滤 1000 个样本的随机信号。

参考下面Matlab代码，给出Python实现代码：


**Matlab代码**
	```
	fc = 300;
	fs = 1000;
	
	[b,a] = cheby1(6,10,fc/(fs/2));
	
	freqz(b,a,[],fs)
	
	subplot(2,1,1)
	ylim([-100 20])
	```


### LLM 生成 ： 

下面这是调用程序
	import numpy as np
	import matplotlib.pyplot as plt
	from scipy.signal import cheby1, freqz
	
	fc = 300
	fs = 1000
	
	b, a = cheby1(6, 10, fc/(fs/2))
	
	w, h = freqz(b, a, fs=fs)
	
	plt.figure()
	plt.subplot(2, 1, 1)
	plt.plot(w, 20 * np.log10(abs(h)))
	plt.ylim(-100, 20)
	plt.show()

## Prompt 2 ： 设计一个归一化边缘频率为0.2π和0.6πrad/sample 和 5 dB 通带纹波。绘制其幅度和相位响应。用它来过滤随机数据。

参考下面Matlab代码，给出Python实现代码：


**Matlab代码**
	```
	[b,a] = cheby1(3,5,[0.2 0.6],'stop');
	freqz(b,a)
	dataIn = randn(1000,1);
	dataOut = filter(b,a,dataIn);
	```


### LLM 生成 ： 

下面这是调用程序
	from scipy.signal import cheby1, freqz, lfilter
	import numpy as np
	
	b, a = cheby1(3, 5, [0.2, 0.6], 'stop')
	w, h = freqz(b, a)
	dataIn = np.random.randn(1000, 1)
	dataOut = lfilter(b, a, dataIn, axis=0)

## Prompt 3 ： 设计一个9阶高通切比雪夫I型滤波器，通带纹波为0.5 dB，通带边缘频率为300 Hz，对于在1000 Hz下采样的数据，对应于0.6πrad/sample。将零点、极点和增益转换为二阶部分。绘制滤波器的幅度和相位响应。

参考下面Matlab代码，给出Python实现代码：


**Matlab代码**
	```
	[z,p,k] = cheby1(9,0.5,300/500,'high');
	sos = zp2sos(z,p,k);
	freqz(sos)
	```


### LLM 生成 ： 

下面这是调用程序
	from scipy.signal import cheby1, zpk2sos, sosfreqz
	
	N = 9
	Wn = 0.5
	b, a = cheby1(N, 0.5, 300/500, 'high')
	sos = zpk2sos(b, a)
	w, h = sosfreqz(sos)

## Prompt 4 ： 设计一个 20 阶切比雪夫 I 型带通滤波器，通带频率较低，通带频率较低，为 400 Hz，通带频率较高，为 560 Hz。 指定 3 dB 的通带纹波和 1500 Hz 的采样率。 使用状态空间表示。

参考下面Matlab代码，给出Python实现代码：


**Matlab代码**
	```
	fs = 1500;
	
	[A,B,C,D] = cheby1(10,3,[400 560]/(fs/2));
	```
### LLM 生成 ： 
下面这是调用程序
	import numpy as np
	from scipy.signal import cheby1
	
	fs = 1500
	
	Wn = np.array([400, 560]) / (fs / 2)
	B, A = cheby1(10, 3, Wn, btype='bandpass')