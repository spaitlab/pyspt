# 信号处理仿真与应用 -数字和模拟滤波器 - 数字滤波器设计

## MATLAB函数描述：cheby2

函数来源：[[切比雪夫II型滤波器设计 - MATLAB cheby2 - MathWorks 中国](https://ww2.mathworks.cn/help/signal/ref/cheby2.html)](https://ww2.mathworks.cn/help/signal/ref/cheby1.html)

### 语法

```matlab
[b,a] = cheby2(n,Rs,Ws)
[b,a] = cheby2(n,Rs,Ws,ftype)
[z,p,k] = cheby2(___)
[A,B,C,D] = cheby2(___)
[___] = cheby2(___,'s')
```

### 说明

[b，a] = cheby2（n，Rs，Ws）返回 th阶的传递函数系数 低通数字切比雪夫II型滤波器，带归一化 阻带边缘频率和分贝 阻带衰减从峰值通带值下降。

[b，a] = cheby2（n，Rs，Ws，ftype）` 设计 低通、高通、带通或带阻切比雪夫 II 型 filter，具体取决于 的值和 的元素数。生成的带通 带阻设计为 2 级。

[z,p,k] = cheby2(___)设计 低通、高通、带通或带阻数字切比雪夫 II 型滤波器，并返回其零点、极点和增益。这 语法可以包含以前语法中的任何输入参数。

[A,B,C,D] = cheby2(___)设计 低通、高通、带通或带阻数字切比雪夫 II 型滤波器，并返回指定其状态空间的矩阵表示法。

[___] = cheby2(___,'s')设计 低通、高通、带通或带阻模拟切比雪夫 II 型滤波器，具有阻带边缘角频率 [`Ws`](https://ww2.mathworks.cn/help/signal/ref/cheby2.html#bucr0qj-Ws) 和 [`Rs`](https://ww2.mathworks.cn/help/signal/ref/cheby2.html#bucr0qj_sep_shared-Rs) 分贝 阻带衰减。

输入参数

### n— 筛选顺序 

整数标量

筛选器顺序，指定为整数标量。对于带通和带阻设计，表示滤波器阶数的一半。

数据类型：double

### Rs— 阻带衰减 

正标量

从峰值通带值向下的阻带衰减，额定值 作为以分贝表示的正标量。

如果您的规格 l 采用线性单位，则可以 使用 = –20 log 将其转换为分贝Rs

**数据类型：**double

### Ws— 阻带边缘频率 标量 |二元素向量

阻带边缘频率，指定为标量或双元素向量。阻带边缘 频率是滤波器的幅度响应为 –[`Rs`](https://ww2.mathworks.cn/help/signal/ref/cheby2.html#bucr0qj_sep_shared-Rs) 分贝的频率。阻带衰减值越大，转换带越宽。`Rs`

- 如果是标量，则设计低通或高通 具有边缘频率的滤波器。`Ws``cheby2``Ws`

  if 是二元素向量，其中 < ， 然后设计带通或带阻 具有较低边沿频率的滤波器和 更高的边缘频率。`Ws``[w1 w2]``w1``w2``cheby2``w1``w2`

- 对于数字滤波器，阻带边缘频率必须位于 在 0 和 1 之间，其中 1 对应奈奎斯特 速率 - 采样速率的一半或 *π* rad/样本。

  对于模拟滤波器，阻带边缘频率必须为 以弧度每秒表示，可以采取任何正数 价值。

**数据类型：**`double`

### ftype— 过滤器类型

筛选器类型，指定为以下类型之一：

- `'low'`指定低通滤波器 阻带边沿频率 [`Ws`](https://ww2.mathworks.cn/help/signal/ref/cheby2.html#bucr0qj-Ws)。 是 标量 的默认值。``
- `'high'`指定高通滤波器 具有阻带边缘频率。
- `'bandpass'`指定带通 如果 为 ，则为 2[`N`](https://ww2.mathworks.cn/help/signal/ref/cheby2.html#bucr0qj_sep_shared-n) 阶的滤波器 一个双元素向量。 是默认值 when 有两个元素。
- `'stop'`指定带阻滤波器 如果 2 阶是双元素向量

## 输出参数

- `b,a`— 传递函数系数行 向量

  

  滤波器的传递函数系数，对于低通和高通滤波器，作为长度为 [`n`](https://ww2.mathworks.cn/help/signal/ref/cheby2.html#bucr0qj_sep_shared-n) + 1 的行向量返回，对于带通和带阻滤波器，返回为长度为 2 + 1 的行向量。对于数字滤波器，传递函数用 和 表示为
  
  ![image-20240430111131375](C:\Users\140104GX\AppData\Roaming\Typora\typora-user-images\image-20240430111131375.png)
  
  对于模拟滤波器，传递函数用 和 表示为
  
  ![image-20240430111150022](C:\Users\140104GX\AppData\Roaming\Typora\typora-user-images\image-20240430111150022.png)

**数据类型：**`double`

### z,p,k— 零点、极点和增益 

列向量，标量

滤波器的零点、极点和增益返回为长度[`为 n`](https://ww2.mathworks.cn/help/signal/ref/cheby2.html#bucr0qj_sep_shared-n) 的两个列向量（带通和带阻设计为 2）和一个标量。

- 对于数字滤波器，传递函数用 、 和 表示为`z``p``k`

  ![image-20240430110309660](C:\Users\140104GX\AppData\Roaming\Typora\typora-user-images\image-20240430110309660.png)

- 对于模拟滤波器，传递函数用 、 和 表示为`z``p``k`

  <img src="C:\Users\140104GX\AppData\Roaming\Typora\typora-user-images\image-20240430110319862.png" alt="image-20240430110319862" style="zoom:80%;" />

**数据类型：**`double`

### A,B,C,D— 状态空间矩阵矩阵 

筛选器的状态空间表示，以矩阵形式返回。 如果 *m* = [`n`](https://ww2.mathworks.cn/help/signal/ref/cheby1.html#bucqk89_sep_shared-n) 为 低通和高通设计，带通和带阻滤波器*的 M* = 2， 然后是 *m* × *m*，是 *m* × 1，是 1 × *m*，是 1 × 1。

- 对于数字滤波器，状态空间矩阵相关 状态向量 *x*，输入 *u*、 和输出 *y* 

  ![image-20240430110337924](C:\Users\140104GX\AppData\Roaming\Typora\typora-user-images\image-20240430110337924.png)

- 对于模拟滤波器，状态空间矩阵相关 状态向量 *x*，输入 *u*、 和输出 *y* 

  ![image-20240430110350558](C:\Users\140104GX\AppData\Roaming\Typora\typora-user-images\image-20240430110350558.png)

**数据类型：**`double`

## Python函数描述：cheby2

函数来源：scipy.signal.cheby1

### 注意事项
**传递函数语法的数值不稳定性**

通常，使用语法来设计 IIR 筛选器。若要分析或实现筛选器，可以将输出与 .如果使用语法设计筛选器，则可能会遇到数值问题。这些问题是由舍入误差引起的，可能发生低至 4 的舍入误差。

### 函数工作原理
切比雪夫II型滤波器在 阻带中的通带和等纹波。II型 过滤器的滚落速度不如 I 型过滤器， 但没有通带纹波。

`cheby2`使用五步算法：

1. 它找到了低通模拟原型 极点、零点和增益使用函数 [`Cheb2AP`](https://ww2.mathworks.cn/help/signal/ref/cheb2ap.html)。
2. 它转换极点、零点和增益 转换为状态空间形式。
3. 如果需要，它使用状态空间 将低通滤波器转换为带通、高通、 或具有所需频率约束的带阻滤波器。
4. 对于数字滤波器设计，它使用[`双线性`](https://ww2.mathworks.cn/help/signal/ref/bilinear.html)来转换模拟滤波器 通过频率双线性变换进入数字滤波器 预翘曲。仔细调整模拟滤波器的频率和 数字滤波器在 [`Ws`](https://ww2.mathworks.cn/help/signal/ref/cheby2.html#bucr0qj-Ws) 或 和 处具有相同的频率响应幅度。
5. 它转换状态空间过滤器 根据需要返回到传递函数或零极点增益形式。

## Prompt 3 ： 应用 Python cheby2 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：

**Matlab代码**

```
fc = 300;
fs = 1000;

[b,a] = cheby2(6,50,fc/(fs/2));

freqz(b,a,[],fs)

subplot(2,1,1)
ylim([-100 20])
dataIn = randn(1000,1);
dataOut = filter(b,a,dataIn);
```




```python
# 定义参数
fc = 300  # 截止频率
fs = 1000  # 采样频率

# 设计低通切比雪夫II型滤波器
[b, a] = cheby2(6, 50, fc / (fs / 2))

# 计算频率响应
w, h = freqz(b, a, worN=8000, fs=fs)
# 绘制幅度和相位响应
plt.figure(figsize=(12, 6))

# 幅度响应
plt.subplot(1, 2, 1)
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Amplitude Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid()

# 相位响应
plt.subplot(1, 2, 2)
plt.plot(w, np.degrees(np.unwrap(np.angle(h))))
plt.title('Phase Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [radians]')
plt.grid()
plt.show()
```



