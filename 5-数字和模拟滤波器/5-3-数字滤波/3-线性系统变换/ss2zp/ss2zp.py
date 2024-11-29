import scipy as scipy

# from scipy.signal import ss2zpk
'''b = [2, 3, 0]
a = [1, 0.4, 1]
# 使用 tf2zpk 函数计算传递函数的零极点
[z1, p1, k1] = scipy.signal.tf2zpk(b, a)
print("Transfer Function Zeros:", z1)
print("Transfer Function Poles:", p1)
print("Transfer Function Gain:", k1)
# 使用 tf2ss 函数将传递函数转换为状态空间表示
[A, B, C, D] = scipy.signal.tf2ss(b, a)
# 使用 ss2zpk 函数计算状态空间表示的零极点
[z, p, k] = scipy.signal.ss2zpk(A, B, C, D)
print("State Space Zeros:", z)
print("State Space Poles:", p)
print("State Space Gain:", k)
'''
fs = 200e3  # 采样率
fp = 75e3  # 通带频率
Fp_norm = fp / (0.5 * fs)  # Nyquist频率的归一化值
[b, a] = scipy.signal.iirfilter(N=6, Wn=Fp_norm, rp=0.2, btype='highpass', ftype='butter')
print(b)
print(a)

