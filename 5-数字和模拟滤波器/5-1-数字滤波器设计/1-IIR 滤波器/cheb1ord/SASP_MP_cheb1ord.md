# 信号处理仿真与应用 -数字和模拟滤波器 - 数字滤波器设计

## MATLAB函数描述：cheb1ord

函数来源：[[切比雪夫 I 类滤波器顺序 - MATLAB cheb1ord - MathWorks 中国](https://ww2.mathworks.cn/help/signal/ref/cheb1ord.html)](https://ww2.mathworks.cn/help/signal/ref/cheby1.html)

### 语法

```matlab
[n,Wp] = cheb1ord(Wp,Ws,Rp,Rs)
[n,Wp] = cheb1ord(Wp,Ws,Rp,Rs,'s')
```

### 说明

```
[n，Wp] = cheb1ord（Wp，Ws，Rp，Rs） `返回 切比雪夫I型滤波器的最低阶，在通带中损耗不超过dB，在阻带中至少具有dB的衰减。的标量（或向量） 还会返回相应的截止频率。
[n，Wp] = cheb1ord（Wp，Ws，Rp，Rs，'s'） 设计 低通、高通、带通或带阻模拟切比雪夫 I 型滤波器，带截止 角频率。
```

输入参数

Wp— 通带转折（截止）频率 标量 |二元素向量

通带拐角（截止）频率，指定为标量或双元素矢量 值介于 0 和 1 之间（含 0 和 1），其中 1 对应于归一化的奈奎斯特 频率，*π* rad/sample。对于数字滤波器，通带的单位 拐角频率以每个样本的弧度为单位。对于模拟滤波器，通带转折频率 以弧度/秒为单位，通带可以是无限的。

- 如果 和 既是标量又是 < ，则返回 低通滤波器。滤波器的阻带范围为 1，通带范围从 0 到Wp``Ws``Wp``Ws``cheb1ord``Ws``Wp

- 如果 和 既是标量又是 > ，则返回高通滤波器。滤波器的阻带范围为 0 至 ，通带范围为 至 Wp``Ws``Wp``Ws``cheb1ord``Ws``Wp

- 如果 和 都是向量和 指定的间隔包含指定的间隔 by （ < < < ），则返回 带通滤波器。滤波器的阻带范围从 0 到 1 和 1。通带范围到 。`Wp``Ws``Ws``Wp``Ws(1)``Wp(1)``Wp(2)``Ws(2)``cheb1ord``Ws(1)``Ws(2)``Wp(1)``Wp(2)`

- 如果都是向量和 指定的间隔包含指定的间隔 by （ < < < ），则返回 带阻滤波器。滤波器的阻带范围为 。通带范围从 0 到 1 和 到。`Wp``Ws``Wp``Ws``Wp(1)``Ws(1)``Ws(2)``Wp(2)``cheb1ord``Ws(1)``Ws(2)``Wp(1)``Wp(2)`

  使用以下指南指定不同类型的筛选器。

  ![image-20240430112357346](C:\Users\140104GX\AppData\Roaming\Typora\typora-user-images\image-20240430112357346.png)

**数据类型：** |single double

### Ws— 阻带转折频率 标量 |二元素向量

阻带转折频率，指定为标量或带有值的双元素向量 介于 0 和 1 之间，其中 1 对应于归一化的奈奎斯特频率。

对于数字滤波器，阻带转折频率以弧度为单位 样本。

对于模拟滤波器，阻带转折频率以弧度/秒为单位，而阻带可以是无限的。

### Rp— 通带纹波 标量

通带纹波，指定为标量，单位为 dB。

**数据类型：** |single double

### Rs— 阻带衰减 标量

阻带衰减，指定为标量，单位为 dB。

**数据类型：** |single double

## 输出参数

### n— 最低滤波器阶数

整数标量

最低筛选器顺序，以整数标量形式返回。

### Wp— 通带转折频率 

标量 | 二元素向量

通带拐角频率，以标量或二元向量形式返回。使用 输出参数 [`n`](https://ww2.mathworks.cn/help/signal/ref/cheb1ord.html#mw_02d636c0-4564-4791-bcd1-d03360a81cbf) 和 [`cheby1`](https://ww2.mathworks.cn/help/signal/ref/cheby1.html) 函数。

## Python函数描述：cheb1ord

函数来源：scipy.signal.cheb1ord

### 函数工作原理
cheb1ord使用切比雪夫低通滤波器阶数预测公式 在 [[1\]](https://ww2.mathworks.cn/help/signal/ref/cheb1ord.html#mw_a8ad444b-396e-4790-a4be-73e557dfc9bb) 中描述。该函数执行 它在模拟和数字情况下的模拟域计算。对于数字案例， 它将频率参数转换为阶数之前的 *S* 域，并且 固有频率估计过程，然后将它们转换回 *z* 域。

cheb1ord`最初开发低通滤波器原型 将所需滤波器的通带频率转换为 1 rad/s（对于低通或 高通滤波器）或 -1 和 1 rad/s（用于带通或带阻滤波器）。然后它计算 低通滤波器匹配通带所需的阶数和固有频率 在使用函数中的值时准确指定。`cheby1

## Prompt 3 ： 应用 Python cheb1ord 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：

**Matlab代码**

```
Wp = 40/500;
Ws = 150/500;
Rp = 3;
Rs = 60;
[n,Wp] = cheb1ord(Wp,Ws,Rp,Rs)
[b,a] = cheby1(n,Rp,Wp);
freqz(b,a,512,1000) 
title('n = 4 Chebyshev Type I Lowpass Filter')
```

python程序：


```python
# 定义参数
Wp = 40 / 500  # 通带截止频率
Ws = 150 / 500  # 阻带截止频率
Rp = 3  # 通带纹波
Rs = 60  # 阻带衰减

# 计算滤波器的阶数和截止频率
n, Wn = cheb1ord(Wp, Ws, Rp, Rs)

# 设计滤波器
b, a = cheby1(n, Rp, Wn, btype='low', analog=False)

# 计算频率响应
w, h = freqz(b, a, worN=512, fs=1000)
# 绘制频率响应
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('n = {} Chebyshev Type I Lowpass Filter'.format(n))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid()
plt.show()

# 绘制相位响应
plt.subplot(2, 1, 2)
plt.plot(w, np.degrees(np.unwrap(np.angle(h))), 'b')
plt.title('Phase')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [degrees]')
plt.grid()
plt.show()
```



