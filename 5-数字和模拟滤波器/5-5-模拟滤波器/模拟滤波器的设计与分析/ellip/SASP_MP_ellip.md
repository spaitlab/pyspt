# 信号处理仿真与应用 - 模拟滤波器 - 模拟滤波器的设计与分析

## MATLAB函数描述：ellip

函数来源：[MATLAB ellip](https://ww2.mathworks.cn/help/signal/ref/ellip.html)
### 语法

[b,a] = ellip(n,Rp,Rs,Wp)
[b,a] = ellip(n,Rp,Rs,Wp,ftype)
[z,p,k] = ellip(___)
[A,B,C,D] = ellip(___)
[___] = ellip(___,'s')

### 说明
[b，a] = ellip（n，Rp，Rs，Wp） 返回 th阶的传递函数系数 具有归一化通带边沿频率的低通数字椭圆滤波器。 得到的滤波器具有峰峰值分贝 通带纹波和阻带衰减的分贝 从峰值通带值下降。nWpRpRs

[b，a] = 椭圆（n，Rp，Rs，Wp，ftype）设计 低通、高通、带通或带阻椭圆滤波器，具体取决于 关于元素的值和数量 之。由此产生的带通和带阻设计 属于 2 阶。ftypeWpn


[z,p,k] = ellip(___)设计 低通、高通、带通或带阻数字椭圆滤波器 并返回其零点、极点和增益。此语法可以包括任何 以前语法中的输入参数。

[A,B,C,D] = ellip(___)设计 低通、高通、带通或带阻数字椭圆滤波器 并返回指定其状态空间表示的矩阵。

[___] = ellip(___,'s')设计 低通、高通、带通或带阻模拟椭圆滤波器 通带边缘角频率 Wp、Rp 分贝 通带纹波和阻带的 Rs 分贝 衰减。

### 输入参数
n— 筛选顺序
整数标量
筛选器顺序，指定为整数标量。对于带通和带阻设计，表示滤波器阶数的一半。
数据类型：double

Rp— 峰峰值通带纹波 正标量 峰峰值通带纹波，指定为正标量 以分贝表示。如果您的规格 l 采用线性单位，则可以 使用 = 40 log 将其转换为分贝Rp10（（1+l）/（1–l））。
数据类型：double

Rs— 阻带衰减 正标量 从峰值通带值向下的阻带衰减，额定值 作为以分贝表示的正标量。如果您的规格 l 采用线性单位，则可以 使用 = –20 log 将其转换为分贝Rs10ℓ.数据类型：double

Wp— 通带边缘频率
标量 |二元素向量
通带边缘频率，指定为标量或双元素矢量。通带边沿 频率是滤波器的幅度响应为 –Rp 分贝的频率。通带纹波值越小，阻带衰减值越大，转换带越宽。如果是标量，则设计低通或高通 具有边缘频率的滤波器。if 是二元素向量，其中 < ， 然后设计带通或带阻 具有较低边沿频率的滤波器和 更高的边缘频率。Wp[w1 w2]w1w2ellipw1w2对于数字滤波器，通带边缘频率必须位于 在 0 和 1 之间，其中 1 对应奈奎斯特 速率 - 采样速率的一半或 π rad/样本。对于模拟滤波器，通带边缘频率必须为 以弧度每秒表示，可以采取任何正数 价值。
数据类型：double

ftype— 过滤器类型
“低” |“带通” |“高” |“停止”
筛选器类型，指定为以下类型之一：

'low'指定低通滤波器 通带边沿频率 Wp。 是 标量 的默认值。'low'Wp

'high'指定高通滤波器 具有通带边沿频率。Wp

'bandpass'指定带通 如果 为 ，则为 2N 阶的滤波器 一个双元素向量。 是默认值 when 有两个元素。Wp'bandpass'Wp

'stop'指定带阻滤波器 如果 2 阶是双元素 向量。nWp
### 输出参量
b,a— 传递函数系数行
向量
滤波器的传递函数系数，对于低通和高通滤波器，作为长度为 n + 1 的行向量返回，对于带通和带阻滤波器，返回为长度为 2 + 1 的行向量。数据类型：double

z,p,k— 零点、极点和增益 列向量，标量 滤波器的零点、极点和增益返回为长度为 n 的两个列向量（带通和带阻设计为 2）和一个标量。数据类型：double

A,B,C,D— 状态空间矩阵矩阵
筛选器的状态空间表示，以矩阵形式返回。 如果 m = n 为 低通和高通设计，带通和带阻滤波器的 M = 2， 然后是 m × m，是 m × 1，是 1 × m，是 1 × 1。数据类型：double

## Python函数描述：ellip
函数来源：自定义
### ellip函数定义：

	import numpy as np
	from scipy import signal

	def ellip(N, rp, rs, Wn, btype='low', analog=False, output='ba'):
    	return signal.ellip(N, rp, rs, Wn, btype, analog, output)

### 参数
- `N`（int）：筛选器的顺序。
- `rs`（float）：以分贝为单位的阻带中的最小衰减。
- `Wn`（浮点或元组）：临界频率。对于低通和高通滤波器，Wn是标量；对于带通滤波器和带阻滤波器，Wn是长度为2的序列。
- `btype`（str，可选）：筛选器的类型（“低”、“高”、“带”、“停止”）。默认值为“低”。
- `analog`（bool，可选）：当为True时，返回模拟滤波器，否则返回数字滤波器。默认值为False。
- `output`（str，可选）：输出类型：“ba”表示分子/分母，“zpk”表示零/极点/增益，或“sos”表示二阶部分。默认值为“ba”。

### 返回值
返回一个椭圆滤波器

## Prompt 1 ：设计一个具有10 dB通带纹波、50 dB阻带衰减和300 Hz通带边缘频率的六阶低通椭圆滤波器，对于在1000 Hz下采样的数据，对应于0.6πrad/sample。绘制其幅度和相位响应。使用它来过滤 1000 个样本的随机信号。

参考下面Matlab代码，给出Python实现代码：


**Matlab代码**

	```
	fc = 300;
	fs = 1000;
	
	[b,a] = ellip(6,10,50,fc/(fs/2));
	
	freqz(b,a,[],fs)
	
	subplot(2,1,1)
	ylim([-100 20])
	dataIn = randn(1000,1);
	dataOut = filter(b,a,dataIn);
	```


### LLM 生成 ： 

下面这是调用程序

	import numpy as np
	import matplotlib.pyplot as plt
	from scipy import signal
	
	fc = 300
	fs = 1000
	
	b, a = signal.ellip(6, 10, 50, fc/(fs/2))
	
	w, h = signal.freqz(b, a, fs=fs)
	
	plt.subplot(2, 1, 1)
	plt.ylim(-100, 20)
	
	dataIn = np.random.randn(1000, 1)
	dataOut = signal.lfilter(b, a, dataIn.flatten())

## Prompt 2 ： 设计一个归一化边缘频率为0.2π和0.6πrad/sample，5 dB通带纹波和50 dB阻带衰减。绘制其幅度和相位响应。用它来过滤随机数据。

参考下面Matlab代码，给出Python实现代码：


**Matlab代码**

	```
	[b,a] = ellip(3,5,50,[0.2 0.6],'stop');
	freqz(b,a)
	dataIn = randn(1000,1);
	dataOut = filter(b,a,dataIn);
	```


### LLM 生成 ： 

下面这是调用程序

	import numpy as np
	import scipy.signal as signal
	
	b, a = signal.ellip(3, 5, 50, [0.2, 0.6], 'stop')
	w, h = signal.freqz(b, a)
	dataIn = np.random.randn(1000, 1)
	dataOut = signal.lfilter(b, a, dataIn.flatten())

## Prompt 3 ： 设计一个通带边缘频率为 300 Hz 的 6 阶高通椭圆滤波器，对于以 1000 Hz 采样的数据，该滤波器对应于0.6πrad/sample。指定 3 dB 的通带纹波和 50 dB 的阻带衰减。将零点、极点和增益转换为二阶部分。绘制幅度和相位响应。

参考下面Matlab代码，给出Python实现代码：


**Matlab代码**

	```
	[z,p,k] = ellip(6,3,50,300/500,"high");
	sos = zp2sos(z,p,k);
	freqz(sos)
	```

### LLM 生成 ： 

下面这是调用程序

	from scipy import signal
	
	z, p, k = signal.ellip(6, 3, 50, 300/500, 'high')
	sos = signal.zpk2sos(z, p, k)
	w, h = signal.sosfreqz(sos)

## Prompt 4 ： 设计一个 20 阶椭圆带通滤波器，其通带频率较低为 500 Hz，通带频率较高，为 560 Hz。 指定 3 dB 的通带纹波、40 dB 的阻带衰减和 1500 Hz 的采样率。 使用状态空间表示。将状态空间表示转换为二阶截面。可视化频率响应。

参考下面Matlab代码，给出Python实现代码：


**Matlab代码**

	```
	fs = 1500;
	
	[A,B,C,D] = ellip(10,3,40,[500 560]/(fs/2));
	sos = ss2sos(A,B,C,D);
	freqz(sos,[],fs)
	```


### LLM 生成 ： 

下面这是调用程序

	import numpy as np
	from scipy import signal
	
	fs = 1500
	
	N, Wn = signal.ellipord(500/(fs/2), 560/(fs/2), 3, 40)
	b, a = signal.ellip(N, 3, 40, Wn, 'low', analog=False, fs=None)
	sos = signal.tf2sos(b, a)
	w, h = signal.sosfreqz(sos, fs=fs)