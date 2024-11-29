# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波器设计

## MATLAB函数描述：butter 

函数来源：[[巴特沃斯滤波器设计 - MATLAB butter - MathWorks 中国](https://ww2.mathworks.cn/help/signal/ref/butter.html)](https://ww2.mathworks.cn/help/signal/ref/envelope.html)

### 语法

```matlab
b,a] = butter(n,Wn)
[b,a] = butter(n,Wn,ftype)
[z,p,k] = butter(___)
[A,B,C,D] = butter(___)
[___] = butter(___,'s')
```

### 说明

`[b,a] = butter(n,Wn)` 返回归一化截止频率为 `Wn` 的 `n` 阶低通数字巴特沃斯滤波器的传递函数系数。

`[b,a] = butter(n,Wn,ftype)` 用于根据 `ftype` 的值和 `Wn` 的元素数目，设计低通、高通、带通或带阻巴特沃斯滤波器。由此得到的带通和带阻设计的阶数都为 2`n`。

`[z,p,k] = butter(___)` 用于设计低通、高通、带通或带阻数字巴特沃斯滤波器，并返回其零点、极点和增益。此语法可包含上述语法中的任何输入参数。

`[A,B,C,D] = butter(___)` 用于设计低通、高通、带通或带阻数字巴特沃斯滤波器，并返回指定其状态空间表示的矩阵。

`[___] = butter(___,'s')` 用于设计截止角频率为 [`Wn`](https://ww2.mathworks.cn/help/signal/ref/butter.html#bucse3u-Wn) 的低通、高通、带通或带阻模拟巴特沃斯滤波器。

### 输入参数

n — 滤波器阶数

整数标量滤波器阶数，指定为整数标量。对于带通和带阻设计，`n` 表示滤波器阶数的一半。

数据类型:double

Wn— 截止频率

标量 | 二元素向量

截止频率，指定为标量或二元素向量。截止频率是滤波器幅值响应为 1 / √2 时的频率。

如果 `Wn` 是标量，则 `butter` 用于设计截止频率为 `Wn` 的低通或高通滤波器。

如果 `Wn` 是二元素向量 `[w1 w2]`，其中 `w1` < `w2`，则 `butter` 用于设计截止频率下限为 `w1` 且截止频率上限为 `w2` 的带通或带阻滤波器。

对于数字滤波器，截止频率必须介于 0 与 1 之间，其中 1 对应于奈奎斯特速率（即采样率的一半）或 π 弧度/采样点。

对于模拟滤波器，截止频率必须用弧度/秒表示，并且可以取任何正值。

数据类型: double

ftype— 滤波器类型
'low'` | `'bandpass'` | `'high'` | `'stop'

滤波器类型，指定为以下项之一：

'low'` 指定截止频率为 [`Wn`](https://ww2.mathworks.cn/help/signal/ref/butter.html#bucse3u-Wn) 的低通滤波器。`'low'` 是标量 `Wn` 的默认值。

'high'` 指定截止频率为 `Wn` 的高通滤波器。

如果 `Wn` 是二元素向量，则 `'bandpass'` 指定阶数为 2[`n`](https://ww2.mathworks.cn/help/signal/ref/butter.html#bucse3u_sep_shared-n) 的带通滤波器。当 `Wn` 有两个元素时，`'bandpass'` 是默认值。

如果 `Wn` 是二元素向量，则 `'stop'` 指定阶数为 2`n` 的带阻滤波器。

### 输出参量

b,a— 传递函数系数 行向量

滤波器的传递函数系数，对于低通滤波器和高通滤波器，以长度为 [`n`](https://ww2.mathworks.cn/help/signal/ref/butter.html#bucse3u_sep_shared-n) + 1 的行向量形式返回；对于带通滤波器和带阻滤波器，以长度为 2`n` + 1 的行向量形式返回。

- 对于数字滤波器，传递函数用 `b` 和 `a` 表示为

  *H*(*z*)=*B*(*z*)*A*(*z*)=b(1)+b(2) *z*−1+⋯+b(n+1) *z*−*n*a(1)+a(2) *z*−1+⋯+a(n+1) *z*−*n*.

  

- 对于模拟滤波器，传递函数用 `b` 和 `a` 表示为

  *H*(*s*)=*B*(*s*)*A*(*s*)=b(1) *s**n*+b(2) *s**n*−1+⋯+b(n+1)a(1) *s**n*+a(2) *s**n*−1+⋯+a(n+1).

数据类型: double

### `z,p,k` — 零点、极点和增益 列向量、标量



滤波器的零点、极点和增益，以长度为 [`n`](https://ww2.mathworks.cn/help/signal/ref/butter.html#bucse3u_sep_shared-n)（对于带通和带阻设计则为 2`n`）的两个列向量以及标量形式返回。

- 对于数字滤波器，传递函数用 `z`、`p` 和 `k` 表示为

  *H*(*z*)=k(1−z(1) *z*−1) (1−z(2) *z*−1)⋯(1−z(n) *z*−1)(1−p(1) *z*−1) (1−p(2) *z*−1)⋯(1−p(n) *z*−1).

- 对于模拟滤波器，传递函数用 `z`、`p` 和 `k` 表示为

  *H*(*s*)=k(*s*−z(1)) (*s*−z(2))⋯(*s*−z(n))(*s*−p(1)) (*s*−p(2))⋯(*s*−p(n)).

**数据类型:** `double`



A,B,C,D — 状态空间矩阵 矩阵

滤波器的状态空间表示，以矩阵形式返回。如果 m = [`n`](https://ww2.mathworks.cn/help/signal/ref/butter.html#bucse3u_sep_shared-n)（对于低通和高通设计）或 m = 2`n`（对于带通和带阻滤波器），则 `A` 为 m×m，`B` 为 m×1，`C` 为 1×m，而 `D` 为 1×1。

- 对于数字滤波器，状态空间矩阵与状态向量 x、输入 u 和输出 y 存在以下关系

  *x*(*k*+1)*y*(*k*)=A *x*(*k*)+B *u*(*k*)=  C *x*(*k*)+D *u*(*k*).

  

- 对于模拟滤波器，状态空间矩阵通过以下方程将状态向量 x、输入 u 和输出 y 相关

  *˙**x*=A *x*+B *u**y*=C *x*+D *u*.

**数据类型:** double

## Python函数描述：butter

函数来源：scipy.signal.butter

### 函数工作原理
巴特沃斯滤波器的幅值响应在通带内具有最大平坦度，并在整体上呈现单调性。这种平滑是以降低滚降陡度为代价的。对于给定滤波器阶数，椭圆和切比雪夫滤波器通常提供更陡的滚降。

`butter` 使用一个五步算法：

1. 它使用函数 [`buttap`](https://ww2.mathworks.cn/help/signal/ref/buttap.html) 查找低通模拟原型的极点、零点和增益。
2. 它将极点、零点和增益转换为状态空间形式。
3. 如果需要，它使用状态空间变换将低通滤波器转换为具有所需频率约束的带通、高通或带阻滤波器。
4. 对于数字滤波器设计，它使用 [`bilinear`](https://ww2.mathworks.cn/help/signal/ref/bilinear.html) 通过具有频率预修正的双线性变换将模拟滤波器转换为数字滤波器。经过仔细调整频率，模拟滤波器和数字滤波器在 `Wn` 或 `w1` 和 `w2` 处可具有相同的频率响应幅值。
5. 根据需要，它将状态空间滤波器转换回其传递函数或零极点增益形式。

## 扩展功能

C/C++ 代码生成 使用 MATLAB® Coder™ 生成 C 代码和 C++ 代码。


## Prompt 2 ： 应用生成的 Python butter 函数做MATLAB示例对应

```
fc = 300;
fs = 1000;

[b,a] = butter(6,fc/(fs/2));

freqz(b,a,[],fs)

subplot(2,1,1)
ylim([-100 20])
```

### LLM 生成 ： 
```python
fc = 300
fs = 1000
b, a = signal.butter(6, fc / (fs / 2))
w, h = signal.freqz(b, a, fs=fs)
# 绘制频率响应
plt.subplot(2, 1, 1)
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Low-pass Butterworth filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
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



