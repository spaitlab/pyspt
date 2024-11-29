# 信号处理仿真与应用 - 数字和模拟滤波器 - 数字滤波器分析

## MATLAB函数描述：zplane 

函数来源：[MATLAB zplane](https://ww2.mathworks.cn/help/signal/ref/zplane.html)

### 语法

zplane(z,p)
zplane(b,a)
[hz,hp,ht] = zplane(___)
zplane(d)
[vz,vp,vk] = zplane(d)

### 说明

zplane(z,p)在当前绘图窗口中绘制列向量z表示的零点和列向量p表示的极点。符号'o'表示零点，符号'x'表示极点。绘图包括参考单位圆。
如果z和p为矩阵，zplane会对z和p的每一列绘制零极点并用不同颜色区分。
zplane(b,a)b和a为行向量，使用roots找到分子系数为b，分母系数为a的系统传递函数的零点和极点。
[hz,hp,ht] = zplane(___)返回绘制零点的向量句柄hz和绘制极点的向量句柄hp。ht是绘制单位圆和文本的向量句柄，
zplane(d)寻找数字滤波器传递函数d的零极点。传递函数d由designfilt设计工具基于频率响应设置生成，并显示在FVTool中。
[vz,vp,vk] = zplane(d)根据数字滤波器传递函数d返回零点vz，极点vp和增益vk。

### 输入参数

z,p — 零点和极点
列向量 | 矩阵
零点和极点，由列向量或矩阵的形式给出。如果z和p是矩阵，那么zplane会对z和p的每一列绘制零极点并用不同颜色区分。
数据类型: single | double

b,a — 传递函数参数
行向量
传递函数参数，由行向量的形式给出。传递函数由z^-1定义。
数据类型: single | double

### 输出参量

vz,vp,vk — 零点，极点和增益
列向量和标量
滤波器的零点，极点和增益，以列向量和标量的形式返回。



## Python函数描述：zplane

函数来源：自定义

### 函数定义：

def zplane(b, a, filename=None):
    zeros = np.roots(b)
    poles = np.roots(a)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Zero-Pole Plot')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='dotted')
    ax.add_artist(circle)
    ax.scatter(np.real(zeros), np.imag(zeros), marker='o', color='g', label='Zeros')
    ax.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label='Poles')
    max_range = max(np.max(np.abs(zeros)), np.max(np.abs(poles)), 1)  # 找到最大范围，包括单位圆
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_aspect('equal', 'box')
    ax.grid(True, which='both')
    ax.legend()
    if filename:
        plt.savefig(filename)
    plt.show()


这段代码定义了一个`zplane`函数，你可以传入分子系数b和分母系数a，函数会计算这些系数的零点和极点，并绘制它们在复平面上的位置。图中还包含了一个单位圆，帮助观察零点和极点相对于单位圆的位置。

### 参数
- `b,a`: 输入传递函数分子系数b和分母系数a。
- `filename`:文件名，如果输入，如果提供，将图保存为文件。

### 注意事项
- 该函数只适用于输入传递函数系数。

### 函数工作原理
1. 使用 `np.roots` 函数计算输入的零极点。
2. 使用matplotlib函数绘制图像。

### 使用场景
这个函数可以用于计算传递函数零极点并绘图。

### 改进建议
- 可以添加对zpk模型输入的图像绘制。



## Prompt 1 ： 生成 Python zplane 函数

import numpy as np
import matplotlib.pyplot as plt


def zplane(b, a, filename=None):
    """
    绘制数字滤波器的零点和极点图，图形窗口长宽相等，以(0, 0)为中心。

    参数:
    - b: 系统的分子系数。
    - a: 系统的分母系数。
    - filename: 如果提供，将图保存为文件。
    """
    # 计算零点和极点
    zeros = np.roots(b)
    poles = np.roots(a)

    # 设置图的大小和标题，确保图窗为正方形
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Zero-Pole Plot')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')

    # 绘制单位圆
    circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='dotted')
    ax.add_artist(circle)

    # 绘制零点和极点
    ax.scatter(np.real(zeros), np.imag(zeros), marker='o', color='g', label='Zeros')
    ax.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label='Poles')

    # 手动设置坐标轴范围以使原点为中心
    max_range = max(np.max(np.abs(zeros)), np.max(np.abs(poles)), 1)  # 找到最大范围，包括单位圆
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])

    # 设置坐标轴的比例为等比例
    ax.set_aspect('equal', 'box')

    # 添加网格线
    ax.grid(True, which='both')

    # 添加图例
    ax.legend()

    # 保存图像
    if filename:
        plt.savefig(filename)

    # 显示图
    plt.show()


# 示例使用
b = [1, -2, 1]  # 分子系数（零点）
a = [1, -1.5, 0.7]  # 分母系数（极点）
zplane(b, a)

