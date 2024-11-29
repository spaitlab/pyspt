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
