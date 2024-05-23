# 信号处理仿真与应用-数字和模拟滤波器-线性系统变换

## MATLAB函数描述 zp2ss

函数来源：https://ww2.mathworks.cn/help/signal/ref/zp2ss.html

### 语法

[A,B,C,D\] = zp2ss(z,p,k)

### 说明

[A,B,C,D] = zp2ss(z,p,k) 查找状态空间表示
$$
\dot{x} = Ax+Bu
$$

$$
\dot{y} = Cx+Du
$$

使得它等价于一个分解传递函数形式的系统
$$
H(s)=\frac{Z(s)}{P(s)}=k\frac{(s-z_{1})(s-z_{2})...(s-z_{n})}{(s-p_{1})(s-p_{2})...(s-p_{n})}
$$
列向量p指定极点位置，矩阵z指定零点位置，有多少列就有多少输出。每个分子传递函数的增益以向量k表示。A、B、C和D矩阵以控制器规范形式返回。

### 输入参数

- z — 系统的零点
  向量
  系统的零点，用向量表示。零必须是实数或者是复数共轭对。如果某些列的零比其他列少，则可以使用Inf值作为z中的占位符。
  示例: [1(1+1j)/2 (1-1j)/2]
  数据类型: double

- p — 系统的极点
  向量
  系统的极点，用向量表示。极点必须是实数或者是复数共轭对。
  示例: [1 (1+1j)/2 (1-1j)/2]
  数据类型: double

- k — 增益标量
  标量
  系统的标量增益，指定为标量。
  数据类型: double

### 输出参数

- A — 状态矩阵
  矩阵
  状态矩阵，作为矩阵返回。如果系统由n个状态变量描述，那么A是n × n的。
  数据类型: single | double

- B — 输入到状态矩阵
  矩阵
  输入到状态矩阵，作为矩阵返回。如果系统由n个状态变量描述，那么B是n × 1。
  数据类型: single | double

- C — 状态到输出矩阵
  矩阵
  状态到输出矩阵，作为矩阵返回。如果系统有q个输出并且由n个状态变量描述，那么C就是q × n。
  数据类型: single | double

- D — 馈通矩阵
  矩阵
  馈通矩阵，作为矩阵返回。如果系统有q个输出，那么D就是q × 1。
  数据类型: single | double

## Python函数描述：zp2ss

函数来源：https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2ss.html

### 语法

scipy.signal.zpk2ss(z,p,k)

### 说明

用于将一个线性时不变系统（LTI system）从零点-极点-增益（ZPK）的表示方式转换为状态空间（State-Space，SS）的表示形式。

### 输入参数

- `z` (sequence): 表示系统零点的数组或序列。
- `p` (sequence): 表示系统极点的数组或序列。
- `k` (float): 系统增益。

### 返回值

- `A` (ndarray): 状态空间表示中的系统矩阵（state matrix），描述了系统中各个状态的相互关系以及各个状态随时间的变化情况。
- `B` (ndarray): 输入矩阵（input matrix），描述了外部输入如何影响系统状态的变化。
- `C` (ndarray): 输出矩阵（output matrix），描述了系统状态如何转化为输出。
- `D` (ndarray): 直传矩阵（feedthrough matrix），它直接连接输入与输出，表示输入对输出的直接影响，在不少系统中这个矩阵常常是零矩阵。

### 注意事项

- **函数名称不同：** 在Python中，函数名是`zpk2ss`，而不是MATLAB中的`zp2ss`。
- **参数类型：** 在Python中使用`zpk2ss`时，零点、极点和增益参数必须是数组或类数组结构，例如列表、元组或NumPy数组。
- **参数顺序：** 函数的参数顺序应该正确；在MATLAB和Python中，这个顺序通常是相同的，但需要确认，尤其是在从MATLAB迁移到Python编程时。
- **输出差异：** 虽然两种语言的函数都转换为状态空间模型，但具体的状态矩阵`A`、`B`、`C`、`D`可能会有区别，因为状态空间表达不是唯一的，可能依赖于内部算法的差异。
- **数值稳定性：** 数值计算稳定性是重要的考虑因素，转换过程可能受到数值问题的影响，特别是当系统的阶数较高或系数接近极限时。

### 使用场景

- **控制系统分析和设计：** 状态空间模型在现代控制理论中非常常见，通常用于分析和设计控制器。
- **系统动态模拟：** 对于复杂系统，状态空间表示可以更加直观地模拟其动态特性。
- **理解内部动态：** 状态空间模型可以帮助理解系统内部的动态过程，特别是当零点和极点不容易提供直观理解时。

### 改进建议

- **清晰的文档：** 对于`zpk2ss`这类函数，提供详细的文档来说明它在不同情况下的行为，特别是关于数值稳定性的讨论。
- **内部算法的透明性：** 解释Python中`zpk2ss`函数中使用的算法和MATLAB可能存在的差异，以帮助用户理解两者之间的结果差异。
- **更多的输出格式选项：** 提供不同的状态变量表示法选项，以便用户可以根据自己的喜好选择最适合自己需求的格式。
- **工具集成：** 在Python中，`scipy`库可以与其他数据分析和可视化库整合，例如`matplotlib`进行图形化展示，`numpy`进行数学运算，这样可以更全面地支持控制系统的设计和分析流程。

## Prompt1：应用python的zpk2ss函数做MATLAB示例对应

**matlab代码**

```matlab
hpFilt = designfilt('highpassiir','FilterOrder',8, ...
         'PassbandFrequency',75e3,'PassbandRipple',0.2, ...
         'SampleRate',200e3);
[z,p,k] = zpk(hpFilt)
```

### LLM 生成 

```python
from scipy import signal
import numpy as np
from pprint import pprint

# 设置numpy打印选项以提高输出的可读性
np.set_printoptions(precision=3, suppress=True)

# 定义零点
z = np.array([0, 0])

# 定义多项式系数并计算极点
p = np.roots([1, 0.01, 1])
print(p)
# 定义增益
k = 1

# 将ZPK表示转换为状态空间表示
A, B, C, D = signal.zpk2ss(z, p, k)

# 美化打印矩阵A, B, C, D
print("矩阵 A:")
pprint(A)
print("\n矩阵 B:")
pprint(B)
print("\n矩阵 C:")
pprint(C)
print("\n矩阵 D:")
pprint(D)
```



