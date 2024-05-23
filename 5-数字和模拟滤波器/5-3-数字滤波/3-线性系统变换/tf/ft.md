

# 信号处理仿真与应用-数字和模拟滤波器-线性系统变换

## MATLAB函数描述：tf

函数来源：[MATLAB tf](https://ww2.mathworks.cn/help/signal/ref/tf.html)

### 语法

[num,den]=tf(d)

### 说明

[num,den]=tf(d)将数字滤波器d转换为分子和分母向量。

### 输入参数

d — 数字滤波器

digitalFilter对象

数字滤波器，指定为digitalFilter对象。使用designfilt根据频率响应设定生成数字滤波器。

示例：d=designfilt（‘lowpassiir’，‘FilterOrder’，3，‘HalfPowerFrequency’，0.5）用于指定归一化3dB频率为0.5π弧度/采样点的三阶巴特沃斯滤波器。

### 输出参量

num — 分子系数

行向量

分子系数，以行向量形式返回。

数据类型：double

den — 分母系数

行向量

分母系数，以行向量形式返回。

数据类型：double

## python函数描述：tf

函数来源：自定义

### 函数定义

```python
def tf(d):
    """
    将数字滤波器对象转换为分子和分母系数向量。
    参数：
    d : 数字滤波器对象，
    # 设定数字滤波器d，例如三阶巴特沃斯滤波器
	# d = designfilt('lowpassiir', 'FilterOrder', 3, 'HalfPowerFrequency', 0.5)
	# NameError: name 'designfilt' is not defined
	# designfilt是matlab中的函数，有同学做了其相对于python的函数
    返回：
    num : array
        分子系数向量。
    den : array
        分母系数向量。
    """
    # 获取数字滤波器的分子和分母系数
    num, den = d.num, d.den

    # 将系数转换为数组形式
    num_arr = np.asarray(num)
    den_arr = np.asarray(den)

    return num_arr, den_arr
```

### 函数定义

## prompt 1 高斯滤波器传递函数

设计一个6阶高通FIR滤波器，其通带频率为75kHz，通带波纹为0.2dB。指定采样频率为200kHz。计算等效传递函数的系数。

参考下面MATLAB的tf函数

```matlab
hpFilt = designfilt('highpassiir','FilterOrder',6, ...
         'PassbandFrequency',75e3,'PassbandRipple',0.2, ...
         'SampleRate',200e3);
[b,a] = tf(hpFilt)
```

实现Python语言高斯滤波器传递函数

```
# `设定数字滤波器d，例如三阶巴特沃斯滤波器`

`d = designfilt('highpassiir','FilterOrder',6, ...`
         `'PassbandFrequency',75e3,'PassbandRipple',0.2, ...`
         `'SampleRate',200e3)`

# `NameError: name 'designfilt' is not defined`

# `designfilt是matlab中的函数，有同学做了其相对于python的函数`

# `调用tf函数将数字滤波器d转换为分子和分母系数`

`num, den = tf(d)`

# `输出分子和分母系数`

`print("分子系数：", num)`
`print("分母系数：", den)`
```





