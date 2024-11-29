# 信号处理仿真与应用 - 模拟滤波器 - 模拟滤波器的设计与分析

## MATLAB函数描述：butter

函数来源：[MATLAB butter](https://ww2.mathworks.cn/help/signal/ref/butter.html)
### 语法

[b,a] = butter(n,Wn)
[b,a] = butter(n,Wn,ftype)
[z,p,k] = butter(___)
[A,B,C,D] = butter(___)
[___] = butter(___,'s')

### 说明
[b,a] = butter(n,Wn) 返回归一化截止频率为 Wn 的 n 阶低通数字巴特沃斯滤波器的传递函数系数。

示例
[b,a] = butter(n,Wn,ftype) 用于根据 ftype 的值和 Wn 的元素数目，设计低通、高通、带通或带阻巴特沃斯滤波器。由此得到的带通和带阻设计的阶数都为 2n。

注意：有关影响传递函数形成的数值问题的信息，请参阅限制。

示例
[z,p,k] = butter(___) 用于设计低通、高通、带通或带阻数字巴特沃斯滤波器，并返回其零点、极点和增益。此语法可包含上述语法中的任何输入参数。

示例
[A,B,C,D] = butter(___) 用于设计低通、高通、带通或带阻数字巴特沃斯滤波器，并返回指定其状态空间表示的矩阵。

示例
[___] = butter(___,'s') 用于设计截止角频率为 Wn 的低通、高通、带通或带阻模拟巴特沃斯滤波器。

### 输入参数
n — 滤波器阶数 整数标量 滤波器阶数，指定为整数标量。对于带通和带阻设计，n 表示滤波器阶数的一半。数据类型: double

Wn — 截止频率 标量 | 二元素向量 截止频率，指定为标量或二元素向量。截止频率是滤波器幅值响应为 1 / √2 时的频率。
如果 Wn 是标量，则 butter 用于设计截止频率为 Wn 的低通或高通滤波器。如果 Wn 是二元素向量 [w1 w2]，其中 w1 < w2，则 butter 用于设计截止频率下限为 w1 且截止频率上限为 w2 的带通或带阻滤波器。

对于数字滤波器，截止频率必须介于 0 与 1 之间，其中 1 对应于奈奎斯特速率（即采样率的一半）或 π 弧度/采样点。
对于模拟滤波器，截止频率必须用弧度/秒表示，并且可以取任何正值。数据类型: double

ftype — 滤波器类型
'low' | 'bandpass' | 'high' | 'stop'
滤波器类型，指定为以下项之一：

'low' 指定截止频率为 Wn 的低通滤波器。'low' 是标量 Wn 的默认值。

'high' 指定截止频率为 Wn 的高通滤波器。

如果 Wn 是二元素向量，则 'bandpass' 指定阶数为 2n 的带通滤波器。当 Wn 有两个元素时，'bandpass' 是默认值。
如果 Wn 是二元素向量，则 'stop' 指定阶数为 2n 的带阻滤波器。

### 输出参量
b,a — 传递函数系数 行向量 滤波器的传递函数系数，对于低通滤波器和高通滤波器，以长度为 n + 1 的行向量形式返回；对于带通滤波器和带阻滤波器，以长度为 2n + 1 的行向量形式返回。数据类型: double

z,p,k — 零点、极点和增益 列向量、标量 滤波器的零点、极点和增益，以长度为 n（对于带通和带阻设计则为 2n）的两个列向量以及标量形式返回。数据类型: double

A,B,C,D — 状态空间矩阵 矩阵 滤波器的状态空间表示，以矩阵形式返回。如果 m = n（对于低通和高通设计）或 m = 2n（对于带通和带阻滤波器），则 A 为 m×m，B 为 m×1，C 为 1×m，而 D 为 1×1。数据类型: double

## Python函数描述：butter
函数来源：自定义
### butter函数定义：

# 定义butter函数
	#定义一个Python函数来模拟matlab中的butter函数
	import scipy.signal

	def butter(n, Wn, btype='low', analog=False, 	output='ba', fs=None):
    	return scipy.signal.butter(n, Wn, btype=btype, analog=analog, output=output, fs=fs)

	# 使用方法示例
	# butter(4, 0.2, 'low', False, 'ba', None)
这段代码是用来定义Matlab中的butter函数的。butter函数是用于设计数字Butterworth滤波器的函数之一。在信号处理中，Butterworth滤波器是一种常见的滤波器类型，具有平滑的频率响应特性。通过调用butter函数并传入相应的参数，可以在Matlab中创建一个Butterworth滤波器。这个函数通常用于对信号进行滤波处理，去除噪音或选择特定频率范围的信号。
### 参数
- `n`：滤波器阶数。
- `Wn`：归一化截止频率，如果是数字，则表示数字滤波器；如果是长度为2的序列，则表示模拟滤波器。
- `btype`：滤波器类型，可以是'low'、'high'、'bandpass'、'bandstop'。
- `analog`：是否为模拟滤波器，默认为False
- `output`：输出类型，可以是'ba'（默认，返回分子和分母多项式的系数）或'sos'（返回二阶节的系数）。
- `fs`：采样频率，用于数字滤波器设计。
### 返回值
函数返回Butterworth滤波器的系统函数系数。


## Prompt 1 ： 设计截止频率为 300 Hz 的 6 阶低通巴特沃斯滤波器，对于以 1000 Hz 的速率采样的数据，截止频率对应于 0.6π 弧度/采样点。绘制其幅值和相位响应。用它对长度为 1000 个采样的随机信号进行滤波。
参考下面MATLAB代码的butter函数
	
	```
	fc = 300;
	fs = 1000;

	[b,a] = butter(6,fc/(fs/2));

	freqz(b,a,[],fs)

	subplot(2,1,1)
	ylim([-100 20])
	```
完成Python语言butter函数的编写，便于类似应用调用。

### LLM 生成 ： 设计截止频率为 300 Hz 的 6 阶低通巴特沃斯滤波器，对于以 1000 Hz 的速率采样的数据，截止频率对应于 0.6π 弧度/采样点。绘制其幅值和相位响应。用它对长度为 1000 个采样的随机信号进行滤波。
	import numpy as np
	import matplotlib.pyplot as plt
	from scipy import signal
	
	fc = 300
	fs = 1000
	
	b, a = signal.butter(6, fc/(fs/2))
	
	w, h = signal.freqz(b, a, fs=fs)
	
	plt.subplot(2, 1, 1)
	plt.plot(w, 20 * np.log10(abs(h)))
	plt.ylim(-100, 20)
	plt.show()


## Prompt 2 ：设计一个 6 阶带阻巴特沃斯滤波器，其归一化边缘频率为 0.2π 和 0.6π 弧度/采样点。绘制其幅值和相位响应。用它对随机数据进行滤波。
	```
	[b,a] = butter(3,[0.2 0.6],'stop');
	freqz(b,a)
	```

### LLM 生成:设计一个 6 阶带阻巴特沃斯滤波器，其归一化边缘频率为 0.2π 和 0.6π 弧度/采样点。绘制其幅值和相位响应。用它对随机数据进行滤波。

	import numpy as np
	import matplotlib.pyplot as plt
	from scipy import signal
	
	b, a = signal.butter(3, [0.2, 0.6], 'bandstop')
	w, h = signal.freqz(b, a)
	plt.plot(w, 20 * np.log10(abs(h)))
	plt.title('Butterworth filter frequency response')
	plt.xlabel('Frequency [radians / second]')
	plt.ylabel('Amplitude [dB]')
	plt.margins(0, 0.1)
	plt.grid(which='both', axis='both')
	plt.show()

## Prompt 3 ：设计一个 9 阶高通巴特沃斯滤波器。指定截止频率为 300 Hz，对于以 1000 Hz 的速率采样的数据，截止频率对应于 0.6π 弧度/采样点。绘制幅值和相位响应。将零点、极点和增益转换为二阶节。显示滤波器的频率响应。
	```
	[z,p,k] = butter(9,300/500,"high");
	sos = zp2sos(z,p,k);
	freqz(sos)
	```

### LLM 生成:设计一个 9 阶高通巴特沃斯滤波器。指定截止频率为 300 Hz，对于以 1000 Hz 的速率采样的数据，截止频率对应于 0.6π 弧度/采样点。绘制幅值和相位响应。将零点、极点和增益转换为二阶节。显示滤波器的频率响应。

	from scipy import signal
	
	b, a = signal.butter(9, 300/500, 'high')
	sos = signal.butter(9, 300/500, 'high', output='sos')
	w, h = signal.sosfreqz(sos)

## Prompt 4 ：设计一个 20 阶带通巴特沃斯滤波器，其截止频率下限为 500 Hz，截止频率上限为 560 Hz。指定采样率为 1500 Hz。使用状态空间表示。使用 designfilt 设计一个相同的滤波器。
	```
	fs = 1500;
	[A,B,C,D] = butter(10,[500 560]/(fs/2));

	d = designfilt("bandpassiir",FilterOrder=20, ...
    HalfPowerFrequency1=500,HalfPowerFrequency2=560, ...
    SampleRate=fs);
	```

### LLM 生成:设计一个 20 阶带通巴特沃斯滤波器，其截止频率下限为 500 Hz，截止频率上限为 560 Hz。指定采样率为 1500 Hz。使用状态空间表示。使用 designfilt 设计一个相同的滤波器。
	
	import numpy as np
	from scipy import signal
	
	fs = 1500
	b, a = signal.butter(10, [500, 560], btype='band', fs=fs)