import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

Fs = 5
dt = 1 / Fs
N = 50
t = np.arange(0, N) * dt
u = np.concatenate(([1], np.zeros(N - 1)))
bf = [1, -(1 + np.cos(dt)), np.cos(dt)]
af = [1, -2 * np.cos(dt), 1]
yf = signal.lfilter(bf, af, u)

plt.plot(t, yf, 'ob-')
plt.plot()
plt.xlabel('t')

A, B, C, D = signal.tf2ss(bf, af)
B = np.atleast_2d(B)
C = np.atleast_2d(C)

x = np.zeros((2, 1))
y = np.zeros(N)
for k in range(N):
    y[k] = np.dot(C, x) + D * u[k]
    x = np.dot(A, x) + B * u[k]

plt.stem(t, y, linefmt='r-', markerfmt='r*', basefmt='') # 添加了空字符串作为基线格式
plt.legend(['tf', 'ss'])
plt.show()