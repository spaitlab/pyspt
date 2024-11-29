# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：fircls 

函数来源：[MATLAB fircls]([Constrained-least-squares FIR multiband filter design - MATLAB fircls - MathWorks 中国](https://ww2.mathworks.cn/help/signal/ref/fircls.html))

### 语法

```
b = fircls(n,f,amp,up,lo)
fircls(n,f,amp,up,lo,"design_flag")
```

### 说明

b = fircls（n，f，amp，up，lo） 生成长度 + 1 线性相位 FIR 滤波器。这 该滤波器的频率-幅度特性与向量 和 给出的频率-幅度特性相匹配。 并且是长度与 相同的向量。他们 定义每个频段中频率响应的上限和下限。
fircls（n，f，amp，up，lo，“design_flag”） 使您能够为滤波器设计指定可视化显示选项。

### 输入参数

n— 滤波顺序
实正标量
滤波器顺序，指定为实数正标量。

f— 归一化频率点
实值向量
归一化频率点，指定为实值向量。过渡 频率在 [0，1] 范围内，其中 1 对应于奈奎斯特频率。这 必须是的第一点和最后一点必须是 。频率必须按递增顺序排列。

amp— 分段常数所需振幅
实值向量
频率响应的分段恒定所需幅度，指定为实值向量。长度等于响应中的波段。

up— 上界
实值向量
每个频段中频率响应的上限，指定为实值 与 amp 长度相同的向量。

lo— 下界
实值向量
每个频带中频率响应的下限，指定为实值 与 amp 长度相同的向量。

"design_flag"— 滤波器设计显示
滤波器设计显示，指定为以下之一：
"trace"— 查看每个设计错误的文本显示 迭代步骤。
"plots"— 查看显示全波段的图集 滤波器的幅度响应以及每个滤波器中幅度响应的缩放视图 子带。所有绘图在每个迭代步骤中都会更新。图上的 O 是 新迭代的估计极值和 X 是 上一次迭代，其中极值是 过滤涟漪。只有具有相应 O 和 X 的涟漪才会产生 平等。
"both"- 同时查看文本显示和绘图。

### 输出参量

b— 滤波器系数
行向量
滤波器系数，以长度为 n + 的行向量返回 1。



## Python函数描述：envelope

函数来源：自定义

## Prompt 1 ： 生成 Python fircls对应示例

参考下面MATLAB代码的envelope函数
```
n = 150;
f = [0 0.4 1];
a = [1 0];
up = [1.02 0.01];
lo = [0.98 -0.01];
b = fircls(n,f,a,up,lo,"both");
```

请使用python语言完成fircls函数对应示例的编写。

### LLM 生成 ： Python fircls 函数示例

    from scipy import signal
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 滤波器的参数
    n = 55  # 滤波器阶数
    f1 = 0.3  # 归一化截止频率
    dp = 0.02  # 通带纹波
    ds = 0.008  # 阻带纹波
    
    # fircls1 在 scipy 中没有直接的等价，但可以使用 remez 来近似
    # fircls1 使用的是 fircls，设置不等的权重来控制通带和阻带波动
    # 计算通带和阻带的权重
    weights = [dp, ds]
    
    # 使用remez设计滤波器
    # 注意：remez函数的bands参数定义了通带和阻带的边界，我们把截止频率分为两部分来指定它们
    # 这里的频带设置需要根据实际情况可能需要进一步调整以匹配MATLAB的结果
    # 还需要考虑MATLAB和Python的频率归一化方式可能不同，这可能导致设计的差异
    b = signal.remez(n+1, [0, f1-dp, f1+ds, 0.5], [1, 0], weight=weights)
    
    # 计算频率响应
    w, h = signal.freqz(b)
    
    # 绘制频率响应
    plt.figure(figsize=(10, 6))
    
    # 绘制总体频率响应，并添加设计的边界
    plt.subplot(3, 1, 1)
    plt.plot(w/np.pi, np.abs(h))
    for edge in [f1-dp, f1+ds]:
        plt.axvline(edge, color='red', linestyle='--')
    plt.title('Magnitude Response')
    plt.xlabel('Frequency [rad/sample]')
    plt.ylabel('Amplitude')
    plt.grid()
    
    # 通带放大视图
    plt.subplot(3, 1, 2)
    plt.plot(w/np.pi, np.abs(h))
    plt.axis([0, 0.5, 1-dp, 1+dp])
    plt.title('Passband Zoom')
    plt.xlabel('Frequency [rad/sample]')
    plt.ylabel('Amplitude')
    plt.grid()
    
    # 阻带放大视图
    plt.subplot(3, 1, 3)
    plt.plot(w/np.pi, np.abs(h))
    plt.axis([0.5, 1, 0, ds])
    plt.title('Stopband Zoom')
    plt.xlabel('Frequency [rad/sample]')
    plt.ylabel('Amplitude')
    plt.grid()
    
    plt.tight_layout()
    plt.show()


