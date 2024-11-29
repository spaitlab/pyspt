# 信号处理仿真与应用 -数字和模拟滤波器 - 数字滤波器设计

## MATLAB函数描述：buttord

函数来源：[[Butterworth filter order and cutoff frequency - MATLAB buttord - MathWorks 中国](https://ww2.mathworks.cn/help/signal/ref/buttord.html)](https://ww2.mathworks.cn/help/signal/ref/envelope.html)

### 语法

```matlab
[n,Wn] = buttord(Wp,Ws,Rp,Rs)
[n,Wn] = buttord(Wp,Ws,Rp,Rs,'s')
```

### 说明

[n，Wn] = buttord（Wp，Ws，Rp，Rs）` 返回数字巴特沃斯的最低阶 滤波器的通带纹波不超过dB，并且 阻带中衰减最小 dB。 和 分别是 滤波器的通带和阻带边缘频率，从 0 归一化到 1， 其中 1 对应于 *π* rad/sample。标量（或向量） 相应的截止频率，也返回。 要设计 Butterworth 过滤器，请使用输出参数并作为 [butter]的输入。

[n，Wn] = buttord（Wp，Ws，Rp，Rs，'s'） 求模拟巴特沃斯滤波器的最小阶数和截止频率。指定 频率和弧度 第二。通带或阻带可以是无限的。

输入参数

Wp— 通带转折（截止）频率
标量 |二元素向量通带转折（截止）频率，指定为标量或双元件 值介于 0 和 1 之间的向量，其中 1 对应归一化 奈奎斯特频率，*π* rad/样本。

- 如果 [`和 Ws`](https://ww2.mathworks.cn/help/signal/ref/buttord.html#d126e10450) 是 标量和<，然后返回低通滤波器的阶数和截止频率。这 滤波器的阻带范围为 1 通带范围从 0 到 。`Wp``Wp``Ws``buttord``Ws``Wp`
- 如果 和 是 标量和>，然后返回高通滤波器的阶数和截止频率。这 滤波器的阻带范围为0至，通带范围为至至 Wp``Ws``Wp``Ws``buttord``Ws``Wp
- 如果 和 是 向量和 指定的区间都包含 （ < < < ） 指定的区间，然后返回阶数和截止频率 带通滤光片。滤波器的阻带范围为 0 往返 1. 通带范围从 到 。Wp``Ws``Ws``Wp``Ws(1)``Wp(1)``Wp(2)``Ws(2)``buttord``Ws(1)``Ws(2)``Wp(1)``Wp(2)
- 如果 和 是 向量和 指定的间隔都包含 （ < < < ） 指定的间隔，然后返回带阻滤波器的阶数和截止频率。 滤波器的阻带范围为 。通带范围从 0 到 1 和 1。Wp``Ws``Wp``Ws``Wp(1)``Ws(1)``Ws(2)``Wp(2)``buttord``Ws(1)``Ws(2)``Wp(1)``Wp(2)

**数据类型：** |single double

### Ws— 阻带转折频率 

标量 |二元素向量

阻带转折频率，指定为标量或双元素向量 值介于 0 和 1 之间，其中 1 对应于归一化的奈奎斯特 频率，*π* rad/sample。

**数据类型：** |single double

### Rp— 通带纹波 

标量

通带纹波，指定为标量，单位为dB。

**数据类型：** |single double

### `Rs`— 阻带衰减 

标量

阻带衰减，指定为标量，单位为dB。

**数据类型：** |`single``double`

## 输出参数





### `n`— 最低滤波器阶 数整数标量



最低筛选器顺序，以整数标量形式返回。



### Wn— 截止频率

标量 | 矢量

截止频率，以标量或向量形式返回。



## Python函数描述：buttord

函数来源：scipy.signal.buttord

### 注意事项
该函数并不适用于非对称信号的真正上下包络线的计算。对于非对称信号，需要更高级的技术，如经验模态分解（Empirical Mode Decomposition, EMD）来获取真正的上下包络线。

### 函数工作原理
buttord的阶次预测公式在模拟域中运算 适用于模拟和数字案例。对于数字外壳，它转换频率 参数添加到 *S* 域之前估计顺序和自然 频率。然后，该函数转换回 *z* 域。

### 使用场景
buttord最初开发低通滤波器原型 将所需滤波器的通带频率转换为 1 rad/秒（对于低通 和高通滤波器）和 –1 和 1 rad/秒（用于带通和带阻滤波器）。它 然后计算低通滤波器满足阻带所需的最小阶数 规范。

## Prompt 3 ： 应用生成的 Python buttord 函数做MATLAB示例对应

参考下面Matlab代码，给出Python实现代码：

**Matlab代码**

```
% 生成高斯调制二次 chirp。指定采样率为 2 kHz，信号持续时间为 2 秒。
t = 0:1/2000:2-1/2000;
q = chirp(t-2,4,1/2,6,'quadratic',100,'convex').*exp(-4*(t-1).^2);
plot(t,q)
% 使用解析信号计算啁啾信号的上下包络。
[up,lo] = envelope(q);
hold on
plot(t,up,t,lo,'linewidth',1.5)
legend('q','up','lo')
hold off
% 由于均值非零，因此，信号不对称。
% 使用不带输出参量的 envelope 以采样编号为 x 轴绘制信号和包络。
envelope(q) 
```


### LLM 生成 ： 

下面这是调用程序
```python
# 定义采样频率和频率规格
fs = 1000  # 采样频率1000 Hz
Wp = 40 / (fs / 2)  # 通带截止频率归一化到[0, 1]
Ws = 150 / (fs / 2)  # 阻带截止频率归一化到[0, 1]
Rp = 3  # 通带纹波最大值3 dB
As = 60  # 阻带衰减最小值60 dB

# 计算滤波器阶数n和截止频率Wn
n, Wn = buttord(Wp, Ws, Rp, As)

# 设计Butterworth滤波器
b, a = butter(n, Wn, btype='low', analog=False)

# 将传递函数转换为Second-Order Sections (SOS)格式
sos = np.array(butter(n, Wn, btype='low', analog=False, output='sos'))

# 计算频率响应
w, h = sosfreqz(sos, worN=512, fs=fs)
```



