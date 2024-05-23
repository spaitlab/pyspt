# from scipy.signal import butter, sos2tf, convolve
# import numpy as np
#
# # 生成4阶低通Butterworth滤波器
# sos = butter(4, 0.5, output='sos')
#
# # 转换为数字滤波器的分子和分母多项式系数
# b, a = sos2tf(sos)
#
# # 计算numers
# numers = [np.convolve(sos[0, 0:3], sos[1, 0:3]) * sos[0, 3] * sos[1, 3], b]
#
# # 计算denoms
# denoms = [convolve(sos[0, 3:6], sos[1, 3:6]), a]
#
# print("Second-Order Sections:")
# print(sos)
# print("numers:", numers)
# print("denoms:", denoms)

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import sosfilt,lfilter,tf2sos

# Define parameters
Fs = 5
dt = 1 / Fs
N = 50
t = np.arange(0, N) * dt
u = np.concatenate(([1], np.zeros(N - 1)))
bf = [1, -(1 + np.cos(dt)), np.cos(dt)]
af = [1, -2 * np.cos(dt), 1]
yf = lfilter(bf, af, u)

plt.plot(t, yf, 'ob-')
plt.plot()
plt.xlabel('t')

plt.show()
# Convert filter coefficients to second-order sections
sos = tf2sos(bf, af)

# Filter the input signal using second-order sections
yt = sosfilt(sos, u)

# Plot the filtered output
plt.stem(t, yt, linefmt='C1-', markerfmt='o', basefmt='k-')
plt.xlabel('t')
plt.title('Filtered Output using SOS')
plt.show()