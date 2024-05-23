# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：findchangepts

函数来源：[MATLAB findchangepts](https://ww2.mathworks.cn/help/signal/ref/findchangepts.html)

### 语法

ipt = findchangepts(x)
ipt = findchangepts(x,Name,Value)
[ipt,residual] = findchangepts(___)
findchangepts(___)

### 说明

ipt = findchangepts(x) 返回 x 均值变化最显着的索引。
如果 x 是具有 N 个元素的向量，则 findchangepts 将 x 划分为两个区域 x(1:ipt-1) 和 x(ipt:N)，从而最小化每个区域与其局部的残差（平方）误差之和意思是。
如果 x 是 M×N 矩阵，则 findchangepts 将 x 划分为两个区域：x(1:M,1:ipt-1) 和 x(1:M,ipt:N)，返回最小化每个区域与其局部 M 维平均值的残差之和。
ipt = findchangepts(x,Name,Value) 使用名称-值参数指定其他选项。选项包括要报告的变化点数量和要测量的统计数据（而不是平均值）。有关详细信息，请参阅变化点检测。
[ipt,residual] = findchangepts(___) 还返回信号相对于建模变化的残余误差，并结合了之前的任何规范。
不带输出参数的 findchangepts(___) 绘制信号和任何检测到的变化点。

### 输入参数

x — 输入信号
实值向量|实值矩阵
输入信号，指定为实值向量或矩阵。
如果 x 是具有 N 个元素的向量，则 findchangepts 将 x 划分为两个区域 x(1:ipt-1) 和 x(ipt:N)，从而最小化每个区域与局部的残差（平方）误差之和统计中指定的统计值。
如果 x 是 M×N 矩阵，则 findchangepts 将 x 划分为两个区域：x(1:M,1:ipt-1) 和 x(1:M,ipt:N)，返回最小化每个区域与统计中指定的统计量的局部 M 维值的残差之和。
示例： reshape(randn(100,3)+[-3 0 3],1,300) 是一个平均值有两次突变的随机信号。
示例： reshape(randn(100,3).*[1 20 5],1,300) 是一个随机信号，其均方根水平有两次突变。
数据类型：single | double

Name-Value 参数
将可选参数对指定为 Name1=Value1,...,NameN=ValueN，其中 Name 是参数名称，Value 是相应的值。名称-值参数必须出现在其他参数之后，但参数对的顺序并不重要。
示例：MaxNumChanges=3,Statistic="rms",MinDistance=20 查找最多三个点，其中均方根水平的变化最显着，并且这些点之间至少有 20 个样本间隔。
在 R2021a 之前，使用逗号分隔每个名称和值，并将名称括在引号中。
示例： 'MaxNumChanges',3,'Statistic',"rms",'MinDistance',20 查找最多三个点，其中均方根水平的变化最显着，并且这些点至少相隔 20 个样本。

MaxNumChanges - 返回的重要更改的最大数量
1（默认）|整数标量
要返回的重要更改的最大数量，指定为整数标量。找到变化最显着的点后，findchangepts 逐渐放宽其搜索标准，以包含更多变化点，但不超过指定的最大值。如果任何搜索设置返回的值超过最大值，则该函数不会返回任何内容。如果未指定 MaxNumChanges，则该函数返回具有最显着变化的点。您不能同时指定 MinThreshold 和 MaxNumChanges。
示例：findchangepts([0 1 0]) 返回第二个样本的索引。
示例：findchangepts([0 1 0],MaxNumChanges=1) 返回一个空矩阵。
示例： findchangepts([0 1 0],MaxNumChanges=2) 返回第二个和第三个点的索引。
数据类型：single | double

Statistic  - 要检测的更改类型
"mean" (default) | "rms" | "std" | "linear"
要检测的更改类型，指定为下列值之一：
“mean” - 检测平均值的变化。如果调用不带输出参数的 findchangepts，该函数将绘制信号、变化点以及由连续变化点包围的每个段的平均值。
"rms" — 检测均方根电平的变化。如果调用不带输出参数的 findchangepts，该函数将绘制信号和变化点。
"std" — 使用高斯对数似然检测标准差的变化。如果调用不带输出参数的 findchangepts，该函数将绘制信号、变化点以及由连续变化点包围的每个段的平均值。
“线性” - 检测平均值和斜率的变化。如果调用不带输出参数的 findchangepts，该函数将绘制信号、变化点以及最适合连续变化点所包围的信号各部分的线。
示例： findchangepts([0 1 2 1],Statistic="mean") 返回第二个样本的索引。
示例： findchangepts([0 1 2 1],Statistic="rms") 返回第三个样本的索引。

MinDistance - 变化点之间的最小样本数
整数标量
变化点之间的最小样本数，指定为整数标量。如果您不指定此数字，则平均值变化的默认值是 1，其他变化的默认值是 2。
示例： findchangepts(sin(2*pi*(0:10)/5),MaxNumChanges=5,MinDistance=1) 返回五个索引。
示例： findchangepts(sin(2*pi*(0:10)/5),MaxNumChanges=5,MinDistance=3) 返回两个索引。
示例： findchangepts(sin(2*pi*(0:10)/5),MaxNumChanges=5,MinDistance=5) 不返回任何索引。
数据类型：single | double


MinThreshold - 总残差误差的最小改进
实标量
每个变化点的总残差误差的最小改进，指定为代表惩罚的实标量。此选项通过对每个预期变更点应用额外惩罚来限制返回的重大变更的数量。您不能同时指定 MinThreshold 和 MaxNumChanges。
示例：findchangepts([0 1 2],MinThreshold=0) 返回两个索引。
示例：findchangepts([0 1 2],MinThreshold=1) 返回一个索引。
示例： findchangepts([0 1 2],MinThreshold=2) 不返回任何索引。
数据类型：single | double



### 输出参量

ipt — 变化点位置
向量
变化点位置，以整数索引向量形式返回。

残差 - 残差
向量
信号相对于建模变化的残余误差，以向量形式返回。

## Python函数描述：findchangepts

函数来源：自定义

### 变化点检测函数定义：

def findchangepts(signal,n_bkps):
    # 使用Binseg方法进行变化点检测
    mode = "l2"  # 使用L2损失，也称为最小均方误差
    algo = rpt.Binseg(model=mode).fit(signal)
    my_bkps = algo.predict(n_bkps=n_bkps-1)
    return my_bkps



这段代码定义了一个名为 `findchangepts` 的函数，其目的是通过计算信号互相关找到信号延迟。

### 参数
- `signal1`: 输入信号，代表时间序列信号。
- `n_bkps`: 重要更改的最大数量。


### 返回值
- `my_bkps`: 变化点位置，以整数索引向量形式返回。

### 注意事项
- 该函数并不适用矩阵计算，只适用于一维信号延迟计算。

### 函数工作原理
1. 使用 `rpt.Binseg` 函数对输入信号进行变化点检测。

### 使用场景
用于对输入信号进行变化点检测。

### 改进建议
- 增加对矩阵信号进行变化点检测的支持。


## Prompt 1 ： 生成 Python findchangepts 函数

参考下面Matlab代码，给出Python实现代码：

**Matlab代码**
```
% 重置随机数生成器以获得可重现的结果。生成随机信号，其中：
% 七个区域中每个区域的平均值都是恒定的，并且各个区域的平均值会突然变化。
% 五个区域中的每一个区域的方差都是恒定的，并且区域之间的方差突然变化。
rng("default")

lr = 20;

mns = [0 1 4 -5 2 0 1];
nm = length(mns);

vrs = [1 4 6 1 3];
nv = length(vrs);

v = randn(1,lr*nm*nv)/2;

f = reshape(repmat(mns,lr*nv,1),1,lr*nm*nv);
y = reshape(repmat(vrs,lr*nm,1),1,lr*nm*nv);

t = v.*y+f;

subplot(2,2,1)
plot(v)
title("Original")
xlim([0 700])

subplot(2,2,2)
plot([f;v+f]')
title("Means")
xlim([0 700])

subplot(2,2,3)
plot([y;v.*y]')
title("Variances")
xlim([0 700])

subplot(2,2,4)
plot(t)
title("Final")
xlim([0 700])
```
请完成Python语言findchangepts函数的编写，便于类似应用调用。


### LLM 生成 ： 

下面这是Python语言findchangepts函数
```
import ruptures as rpt

def findchangepts(signal,n_bkps):
    # 使用Binseg方法进行变化点检测
    mode = "l2"  # 使用L2损失，也称为最小均方误差
    algo = rpt.Binseg(model=mode).fit(signal)
    my_bkps = algo.predict(n_bkps=n_bkps-1)
    return my_bkps

import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，确保结果的可重复性
np.random.seed(0)

# 定义常数
lr = 20

# 平均值和方差
mns = np.array([0, 1, 4, -5, 2, 0, 1])
vrs = np.array([1, 4, 6, 1, 3])

# 计算区域数量
nm = len(mns)
nv = len(vrs)

# 生成原始随机值
v = np.random.randn(lr * nm * nv) / 2


# 构造平均值变化
f = np.repeat(np.array([mns]), lr*nv,axis=0)
f=f.reshape(1,-1,order='F')
f = f.flatten()

 
# 构造方差变化
y = np.repeat(np.array([vrs]), lr*nm,axis=0)
y=y.reshape(1,-1,order='F')  # Ensure the size matches v by repeating for each mean segment
y= y.flatten()


# 计算最终信号
t = v * y + f

# 绘制图形
plt.figure(figsize=(6, 6))

plt.subplot(2, 2, 1)
plt.plot(v)
plt.title("Original")
plt.xlim([0, lr * nm * nv])

plt.subplot(2, 2, 2)
plt.plot(f)
plt.plot(v + f)
plt.title("Means")
plt.xlim([0, lr * nm * nv])

plt.subplot(2, 2, 3)
plt.plot(y)
plt.plot(v * y)
plt.title("Variances")
plt.xlim([0, lr * nm * nv])

plt.subplot(2, 2, 4)
plt.plot(t)
plt.title("Final")
plt.xlim([0, lr * nm * nv])

plt.tight_layout()
plt.show()


import ruptures as rpt
import matplotlib.pyplot as plt
from findchangepts import findchangepts 

# 使用findchangepts进行变化点检测
my_bkps = findchangepts(signal=t,n_bkps=6)
# 绘制数据
plt.plot(t)
print("变化位置：",my_bkps)
my_bkps.insert(0,0)

for i, pos in enumerate(my_bkps[:-1]):  # 遍历x_positions列表除了最后一个元素
    next_pos = my_bkps[i + 1]
    # 在指定位置添加竖线
    if i != 0:
        plt.axvline(x=pos, color='g')

    # 计算当前竖线和下一竖线之间数据点的均值
    mean_value = np.mean(t[pos:next_pos])
    
    # 在两个竖线之间添加表示均值的横线
    plt.hlines(mean_value, pos, next_pos, colors='r')


```