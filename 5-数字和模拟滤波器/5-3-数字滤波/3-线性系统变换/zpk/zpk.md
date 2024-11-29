# 信号处理仿真与应用-数字和模拟滤波器-线性系统变换

## MATLAB函数描述 zpk

函数来源：https://ww2.mathworks.cn/help/signal/ref/zpk.html

### 语法

[z,p,k] = zpk(d)

### 说明

[z,p,k] = zpk(d) 分别以向量z和p以及标量k的形式返回数字滤波器d对应的零点、极点和增益。

### 输入参数

- d — 数字滤波器
  数字滤波器对象
  数字滤波器，指定为数字滤波器对象。可以使用designfilt函数根据频率响应规格生成数字滤波器。
  示例:  d = designfilt('lowpassiir','FilterOrder',3,'HalfPowerFrequency',0.5) 指定一个三阶巴特沃斯滤波器，归一化3db频率为0.5π rad/sample。
  数据类型: double

### 输出参数

- z — 系统的零点
  列向量
  过滤器的零点，作为列向量返回。
  数据类型: double
- p — 系统的极点
  列向量
  过滤器的极点，作为列向量返回。
  数据类型: double

- k — 系统的增益
  实标量
  过滤器的增益，作为实数标量返回。
  数据类型: double

## python的函数描述 scipy.signal.tf2zpk

函数来源：https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2zpk.html

### 语法

z, p, k = scipy.signal.tf2zpk(numerator, denominator)

### 说明

将传递函数的系数转换为零点-极点-增益（ZPK）的形式。传递函数通常以分子和分母多项式系数的形式给出，以描述一个线性时不变系统的输入和输出之间的关系。

### 输入参数

- `numerator`: 分子多项式系数的一维数组，按降序排列。
- `denominator`: 分母多项式系数的一维数组，按降序排列。

### 返回值

- `z`: 系统零点的数组。
- `p`: 系统极点的数组。
- `k`: 系统增益。

### 注意事项

- 输入的b和a系数应当是一维数组，其定义传递函数的多项式系数，顺序从最高次项开始到常数项。
- 在MATLAB中，针对`digitalFilter`对象的`zpk`函数会自动处理滤波器对象和给出滤波器的零点、极点和增益。相比之下，Python的`tf2zpk`需要用户已经得到了传递函数的系数。
- 数字稳定性：在进行转换时，应当确保滤波器的设计是稳定的，因为极点的位置会影响到系统的稳定性。
- 数值精度：`tf2zpk`可能会受到数值精度的影响，尤其是当处理接近系统阶跃响应或倍频滤波时。极点和零点的计算可能会有轻微的差异，特别是当系数非常小时。
- 格式化和展示：与MATLAB的`zpk`函数一样，得到的ZPK结果可以直接用于进一步的分析或可视化，但在展示时可能需要适当的格式化。

### 使用场景

- 在信号处理或数字通信系统设计中，设计滤波器通常是第一步，而将滤波器的传递函数转换为ZPK格式有助于更深入地理解滤波器的性质和响应。
- 用于分析滤波器的稳定性和频率响应。
- 在控制系统中，ZPK格式有助于分析系统的时间响应和稳定性边界。

### 改进建议

- **隐士处理：** Python的`scipy.signal`模块可以在未来的更新中提供类似MATLAB`digitalFilter`对象的包裹器，让用户可以传入一个滤波器对象而不是系数数组，从而更直接地获取ZPK参数。
- **可视化工具：** 与MATLAB一样，Python也可以提供封装好的函数来直观显示滤波器的零点和极点图，使得滤波器的性能变得更加直观。
- **自动格式化：** `tf2zpk`函数可以提供可选的格式化输出，使得零点和极点数据打印出来更易于阅读，尤其是对于非技术背景的用户。

## Prompt1：应用python的scipy.signal.tf2zpk函数做MATLAB示例对应

**matlab代码**

```matlab
hpFilt = designfilt('highpassiir','FilterOrder',8, ...
         'PassbandFrequency',75e3,'PassbandRipple',0.2, ...
         'SampleRate',200e3);
[z,p,k] = zpk(hpFilt)
```

### LLM生成

```python
from scipy.signal import iirfilter, tf2zpk

# 滤波器的规格参数
filter_order = 8
passband_freq = 75e3  # 通带频率
passband_ripple = 0.2  # 通带波动
sample_rate = 200e3    # 采样率

# 设计IIR高通滤波器（Chebyshev类型I）
b, a = iirfilter(N=filter_order, Wn=passband_freq / (0.5 * sample_rate),
                 rp=passband_ripple, btype='high',
                 analog=False, ftype='cheby1', output='ba')

# 使用tf2zpk转换传递函数系数为零点、极点和增益
z, p, k = tf2zpk(b, a)

# 格式化输出函数
def format_complex_number(c):
    real = f"{c.real:.6f}"
    imag = f"{abs(c.imag):.6f}" + "j" if c.imag != 0 else ""
    sign = "+" if c.imag >= 0 else "-"
    return f"({real} {sign} {imag})" if imag else f"({real})"

# 优化后的打印输出
print("Zeros: [", ", ".join(format_complex_number(zi) for zi in z), "]")
print("Poles: [", ", ".join(format_complex_number(pi) for pi in p), "]")
print(f"Gain: {k:.9f}")
```
