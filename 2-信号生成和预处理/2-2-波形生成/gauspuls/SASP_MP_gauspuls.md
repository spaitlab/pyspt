# 信号处理仿真与应用 - 信号生成和预处理 - 波形生成

## MATLAB函数描述：gauspuls

函数来源：[MATLAB envelope](https://ww2.mathworks.cn/help/signal/ref/gauspuls.html)

### 语法

yi = gauspuls(t,fc,bw)
yi = gauspuls(t,fc,bw,bwr)
[yi,yq] = gauspuls(___)
[yi,yq,ye] = gauspuls(___)
tc = gauspuls('cutoff',fc,bw,bwr,tpe)

### 说明

yi = gauspuls(t,fc,bw) 在阵列t中指定的时间返回一个单位振幅高斯调制的正弦射频脉冲，中心频率fc以赫兹为单位，带宽为bw

yi = gauspuls(t,fc,bw,bwr) 返回一个单位振幅的相位高斯RF脉冲，相对于归一化信号峰值，在bwr dB的水平上测量得到bw的分数带宽。

[yi,yq] = gauspuls(___) 也返回正交脉冲。这种语法可以包含前一种语法的输入参数的任意组合。

[yi,yq,ye] = gauspuls(___) 返回射频信号包络。

tc = gauspuls('cutoff',fc,bw,bwr,tpe) 返回截止时间tc，此时尾脉冲包络相对于峰值包络幅度低于tpe dB。

### 输入参数

t — 时间值向量
向量
计算单位幅度高斯RF脉冲的时间值向量。
数据类型: single | double

fc — 中心频率
1000(默认)|实正标量
高斯调制正弦脉冲的中心频率，用实正标量表示，以Hz表示。

bw — 分数带宽
0.5(默认)|实正标量
高斯调制正弦脉冲的分数带宽，用实正标量表示。

bwr — 分数带宽参考电平
-6(默认)|实负标量
高斯调制正弦脉冲的分数带宽参考电平，用实负标量表示。BWR表示比峰值(单位)包络幅度小的参考水平。分数带宽由功率比指定。这对应于以幅度比表示的-3 dB点。

tpe - 拖尾脉冲包络电平
-60(默认)|实负标量
尾脉冲包络电平，用实负标量表示，单位为dB。tpe表示比峰值(单位)包络幅度小的参考水平。

### 输出参量

yi — 异相高斯脉冲
向量
相位高斯调制正弦脉冲，在时间向量t表示的时间以单位振幅向量的形式返回。

yq - 正交高斯脉冲
向量
正交高斯调制正弦脉冲，在时间向量t表示的时间以单位振幅向量的形式返回。

ye - 射频信号包络
向量
在时间向量t所表示的时刻单位幅度的射频信号包络。

tc — 截止时间
正实数标量
相对于峰值包络幅度，尾脉冲包络下降到tpe dB以下的截止时间，单位为秒。

## Python函数描述：gausspulse

函数来源：scipy.signal

### 高斯脉冲函数调用：

scipy.signal.gausspulse(t, fc=1000, bw=0.5, bwr=-6, tpr=-60, retquad=False, retenv=False)

返回一个高斯调制正弦信号:
exp(-a t^2) exp(1j*2*pi*fc*t).
如果retquad为True，则返回实部和虚部(同相和正交)。如果retenv为True，则返回包络(未调制信号)。否则，返回调制正弦信号的实部。

### 参数
- `t`: Ndarray或者字符串' cutoff '。输入数组。
- `fc`: float,可选。中心频率(例如Hz)。默认值是1000。
- `bw`: float,可选。脉冲的频域分数带宽(如Hz)。默认值为0.5。
- `bwr`: float,可选。计算分数带宽的参考电平(dB)。默认值为-6。
- `tpr`: float,可选。如果t是` cutoff `，则该函数返回脉冲幅度低于tpr时的截止时间(以dB为单位)。默认值为-60。
- `retquad`: bool,可选。如果为真，则返回信号的积分(虚数)和实部。默认值为False
- `retenv`: bool,可选。如果为真，返回信号的包络。默认值为False

### 返回值
- `yI`: ndarray。信号的实部。总是返回。
- `yQ`: ndarray。信号的虚部。仅当retquad为True时返回。
- `yenv`: ndarray。信号包络。仅当retenv为True时返回。


## Prompt ： 应用scipy.signal.gausspulse 函数做MATLAB示例对应

```
tc = gauspuls('cutoff',50e3,0.6,[],-40); 
t = -tc : 1e-7 : tc; 
[yi,yq,ye] = gauspuls(t,50e3,0.6); 

plot(t,yi,t,yq,t,ye)
legend('Inphase','Quadrature','Envelope')
```

请给出以上代码的python对应，并绘制图形。

### LLM 生成 ： 
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gausspulse

# 定义参数
fc = 50e3  # 载波频率
bw = 0.6   # 带宽

# 计算截止时间，这里用到的是高斯脉冲的特性，需要手动调整以适应 -40dB 的截止标准
tc = 6 / (np.pi * bw * fc)

# 生成时间数组
t = np.arange(-tc, tc, 1e-7)

# 生成高斯脉冲
yi, yq = gausspulse(t, fc=fc, bw=bw, retquad=True)
ye = np.abs(yi + 1j * yq)  # 计算包络

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(t, yi, label='Inphase')
plt.plot(t, yq, label='Quadrature')
plt.plot(t, ye, label='Envelope')
plt.legend()
plt.title('Gaussian Pulse')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```


