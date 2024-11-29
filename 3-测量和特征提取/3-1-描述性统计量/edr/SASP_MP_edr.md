# 信号处理仿真与应用 - 测量和特征提取 - 描述性统计量

## MATLAB函数描述：edr 

函数来源：[MATLAB edr](https://ww2.mathworks.cn/help/signal/ref/edr.html)

### 语法

dist = edr(x,y,tol)
[dist,ix,iy] = edr(x,y,tol)
[___] = edr(x,y,maxsamp)
[___] = edr(___,metric)
edr(___)

### 说明

dist = edr(x,y,tol) 返回序列 x 和 y 之间真实信号​​的编辑距离。 edr 返回必须从 x、y 或 x 和 y 中删除的元素的最小数量，以便剩余信号元素之间的欧几里得距离之和位于指定的容差 tol 内。
[dist,ix,iy] = edr(x,y,tol) 返回扭曲路径，使得 x(ix) 和 y(iy) 之间的距离尽可能最小。当 x 和 y 是矩阵时，ix 和 iy 使得 x(:,ix) 和 y(:,iy) 最小分离。
[___] = edr(x,y,maxsamp) 限制插入操作，以便扭曲路径保持在 x 和 y 之间直线拟合的 maxsamp 样本内。此语法返回先前语法的任何输出参数。
[___] = edr(___,metric) 指定除了先前语法中的任何输入参数之外要使用的距离度量。 metric 可以是“euclidean”、“absolute”、“squared”或“symmkl”之一。
不带输出参数的 edr(___) 绘制原始信号和对齐信号。
如果信号是实向量，则该函数会在子图中显示两个原始信号，并在第一个信号下方的子图中显示对齐信号。
如果信号是复向量，该函数会在三维图中显示原始信号和对齐信号。
如果信号是实数矩阵，则该函数使用 imagesc 来显示原始信号和对齐信号。
如果信号是复数矩阵，该函数会在每个图像的上半部分和下半部分绘制它们的实部和虚部。

### 输入参数

x - 输入信号
矢量|矩阵
输入信号，指定为实数或复数向量或矩阵。
数据类型：single | double
复数支持：是

y - 输入信号
矢量|矩阵
输入信号，指定为实数或复数向量或矩阵。
数据类型：single | double
复数支持：是

tol——公差
正标量
容差，指定为正标量。
数据类型：single | double

maxsamp - 调整窗口的宽度
Inf（默认）|正整数
调整窗口的宽度，指定为正整数。
数据类型：single | double

metric - 距离度量
'euclidean' (default) | 'absolute' | 'squared' | 'symmkl'
距离度量，指定为“euclidean”、“absolute”、“squared”或“symmkl”。如果 X 和 Y 都是 K 维信号，则度量规定 dmn(X,Y)，即 X 的第 m 个样本与 Y 的第 n 个样本之间的距离。

### 输出参量

dist - 最小距离
正实标量
信号之间的最小距离，以正实标量形式返回。

ix,iy - 扭曲路径
索引向量
扭曲路径，以索引向量形式返回。 ix 和 iy 具有相同的长度。每个向量包含一个单调递增的序列，其中对应信号 x 或 y 的元素的索引重复必要的次数。

## Python函数描述：edr

函数来源：自定义

### edr函数定义：

import numpy as np

def edr(seq1, seq2, tolerance):
    """
    Calculates the Edit Distance on Real sequence (EDR) between two sequences.
    This function assumes that the sequences are lists of real numbers.
    
    :param seq1: First sequence of real numbers
    :param seq2: Second sequence of real numbers
    :param tolerance: Tolerance within which matches are considered
    :return: EDR distance and the aligned sequences
    """
    len_seq1 = len(seq1)
    len_seq2 = len(seq2)
    # Create a 2D matrix of size (len_seq1+1) x (len_seq2+1) for dynamic programming
    dp = np.zeros((len_seq1 + 1, len_seq2 + 1))

    # Initialize the first row and column of the matrix
    for i in range(len_seq1 + 1):
        dp[i][0] = i
    for j in range(len_seq2 + 1):
        dp[0][j] = j

    # Populate the matrix based on the EDR distance rule
    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            if abs(seq1[i - 1] - seq2[j - 1]) <= tolerance:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # Insertion
                                    dp[i][j - 1],  # Deletion
                                    dp[i - 1][j - 1])  # Match/Mismatch

    # Reconstruct the aligned sequences
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = len_seq1, len_seq2
    while i > 0 and j > 0:
        if abs(seq1[i - 1] - seq2[j - 1]) <= tolerance:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(0)
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            aligned_seq1.append(0)
            aligned_seq2.append(seq2[j - 1])
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1

    # Finish the remaining sequence if any
    while i > 0:
        aligned_seq1.append(seq1[i - 1])
        aligned_seq2.append(0)
        i -= 1
    while j > 0:
        aligned_seq1.append(0)
        aligned_seq2.append(seq2[j - 1])
        j -= 1

    aligned_seq1.reverse()
    aligned_seq2.reverse()

    return dp[-1][-1], aligned_seq1, aligned_seq2


这段代码定义了一个名为 `edr` 的函数，用于计算两个序列（seq1和seq2）之间的编辑距离，同时考虑到了一个给定的容差（tolerance）。

### 参数
- `seq1`: 输入信号1，可以是任何数组形式的数据，通常是一维的，代表时间序列信号。
- `seq2`: 输入信号2，可以是任何数组形式的数据，通常是一维的，代表时间序列信号。
- `tolerance`: 容差，两个序列值被视为相等的最大允许差。



### 返回值
- 两个序列之间的编辑距离

### 注意事项

### 函数工作原理

1. 初始化动态规划矩阵 D：使用 NumPy 创建一个大小为 (m+1) x (n+1) 的零矩阵，其中 m 和 n 分别是输入序列的长度。这个矩阵用于存储计算过程中的中间结果。
2. 填充矩阵的第一行和第一列：矩阵的第一行和第一列被初始化以反映从空序列到每个序列达到当前长度的最小编辑次数。例如，从空序列到长度为 i 的序列至少需要 i 次插入操作。
3. 使用动态规划计算编辑距离：通过比较序列元素并根据给定的容差计算成本（cost），逐个填充矩阵的其余部分。如果两个元素的差的绝对值小于或等于容差，则认为这两个元素相等，成本（cost）为 0；否则，成本为 1。矩阵的每个单元格包含从序列 seq1 的前 i 个元素到序列 seq2 的前 j 个元素的最小编辑距离，考虑到插入、删除和替换操作。
4. 返回编辑距离：函数最终返回矩阵右下角的值 D[m, n]，这代表了将整个序列 seq1 转换为整个序列 seq2 所需的最小编辑操作数。

### 使用场景
这个函数可以用于两个信号之间的编辑距离。

### 改进建议
- 增加对矩阵的支持

## Prompt 1 ： 生成 Python edr 函数

参考下面MATLAB代码的edr函数
```
% 生成由两个不同峰组成的两个信号，两个峰被不同长度的谷分隔开。绘制信号图。
x1 = [0 1 0 1 0]*.95;
x2 = [0 1 0 0 0 0 0 0 0 0 1 0]*.95;

subplot(2,1,1)
plot(x1)
xlim([0 12])
subplot(2,1,2)
plot(x2)
xlim([0 12])

% 计算信号之间的编辑距离。设置较小的容差，以便唯一的匹配发生在相同的样本之间。
tol = 0.1;

figure
edr(x1,x2,tol);
```

请完成Python语言edr函数的编写，便于类似应用调用。

### LLM 生成 ： Python edr 函数

这是edr函数定义
```
import numpy as np

def edr(seq1, seq2, tolerance):
    """
    Calculates the Edit Distance on Real sequence (EDR) between two sequences.
    This function assumes that the sequences are lists of real numbers.
    
    :param seq1: First sequence of real numbers
    :param seq2: Second sequence of real numbers
    :param tolerance: Tolerance within which matches are considered
    :return: EDR distance and the aligned sequences
    """
    len_seq1 = len(seq1)
    len_seq2 = len(seq2)
    # Create a 2D matrix of size (len_seq1+1) x (len_seq2+1) for dynamic programming
    dp = np.zeros((len_seq1 + 1, len_seq2 + 1))

    # Initialize the first row and column of the matrix
    for i in range(len_seq1 + 1):
        dp[i][0] = i
    for j in range(len_seq2 + 1):
        dp[0][j] = j

    # Populate the matrix based on the EDR distance rule
    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            if abs(seq1[i - 1] - seq2[j - 1]) <= tolerance:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # Insertion
                                    dp[i][j - 1],  # Deletion
                                    dp[i - 1][j - 1])  # Match/Mismatch

    # Reconstruct the aligned sequences
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = len_seq1, len_seq2
    while i > 0 and j > 0:
        if abs(seq1[i - 1] - seq2[j - 1]) <= tolerance:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(0)
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            aligned_seq1.append(0)
            aligned_seq2.append(seq2[j - 1])
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1

    # Finish the remaining sequence if any
    while i > 0:
        aligned_seq1.append(seq1[i - 1])
        aligned_seq2.append(0)
        i -= 1
    while j > 0:
        aligned_seq1.append(0)
        aligned_seq2.append(seq2[j - 1])
        j -= 1

    aligned_seq1.reverse()
    aligned_seq2.reverse()

    return dp[-1][-1], aligned_seq1, aligned_seq2
```
这是edr函数调用示例
```
import numpy as np
from edr import edr
# Example usage
x1 = np.array([0, 1, 0, 1, 0]) * 0.95
x2 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]) * 0.95
tolerance = 0.1
edr_distance, aligned_x1, aligned_x2 = edr(x1, x2, tolerance)

edr_distance, aligned_x1, aligned_x2


import matplotlib.pyplot as plt

# Use the calculated aligned sequences to plot the result
plt.figure(figsize=(6, 6))

# Plot the original sequences
plt.subplot(2, 1, 1)
plt.plot(x1, label='Sequence 1', marker='o')
plt.plot(x2, label='Sequence 2', marker='x')
plt.title('Original Sequences')
plt.legend()

# Plot the aligned sequences with None values removed
aligned_x1_clean = [val for val in aligned_x1 if val is not None]
aligned_x2_clean = [val for val in aligned_x2 if val is not None]

plt.subplot(2, 1, 2)
plt.plot(aligned_x1_clean, label='Aligned Sequence 1', marker='o')
plt.plot(aligned_x2_clean, label='Aligned Sequence 2', marker='x')
plt.title('Aligned Sequences (Edit Distance: {:.1f})'.format(edr_distance))
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()

```
