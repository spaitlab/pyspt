import numpy as np
import scipy

# 定义一个函数tf，接受数字滤波器d作为输入
def tf(d):
    # 将数字滤波器d转换为分子和分母向量
    num, den = d.num, d.den
    return num, den

# 示例用法
# 设定数字滤波器d，例如三阶巴特沃斯滤波器
d = designfilt('lowpassiir', 'FilterOrder', 3, 'HalfPowerFrequency', 0.5)
#NameError: name 'designfilt' is not defined
# designfilt是matlab中的函数，有同学做了其相对于python的对应
# 调用tf函数将数字滤波器d转换为分子和分母系数
num, den = tf(d)

# 输出分子和分母系数
print("分子系数：", num)
print("分母系数：", den)
