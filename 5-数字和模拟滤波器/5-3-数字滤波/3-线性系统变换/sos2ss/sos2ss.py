from scipy.signal import sos2tf, tf2ss

def sos2ss(sos, g=1):
    # 将SOS表示的数字滤波器转换为传递函数的分子和分母形式
    b, a = sos2tf(sos)

    # 将传递函数乘以增益因子
    b = [g * coef for coef in b]

    # 使用传递函数转换为状态空间方程
    A, B, C, D = tf2ss(b, a)
    return A, B, C, D