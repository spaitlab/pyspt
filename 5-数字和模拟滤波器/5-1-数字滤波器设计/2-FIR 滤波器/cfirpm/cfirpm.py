# import numpy as np
# from scipy.signal import remez

# def cfirpm(numtaps, bands, ftype, fs=2):
#     if ftype == 'lowpass':
#         # Normalize bands to [0, fs/2]
#         bands = np.array(bands)
#         bands = (bands + 1) / 2  # Normalize bands to [0, 1]

#         # Define the desired gain for each band
#         # We have two stop bands and one pass band, resulting in three desired levels.
#         desired = [0, 1, 0]  # Desired gains for stop band, pass band, and stop band

#         # Define weights for each band
#         weight = [1, 1, 1]  # Equal weights for simplicity

#         # Check if the bands array is structured correctly
#         if len(bands) != 2*len(desired):
#             raise ValueError("Length of bands must be twice the length of desired.")

#         # Use remez to calculate filter coefficients
#         taps = remez(numtaps+1, bands, desired, weight, fs=fs)
#         return taps
#     else:
#         raise ValueError("Filter type not supported.")

# # The code below is for testing and should be in a separate file or script.
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from scipy.signal import freqz

#     # Call cfirpm to design filter
#     b = cfirpm(30, [-1, -0.5, -0.4, 0.7, 0.8, 1], 'lowpass')

#     # Compute the frequency response
#     w, h = freqz(b, worN=8192, whole=True)

#     # Plot the magnitude response
#     plt.subplot(2, 1, 1)
#     plt.plot(w / np.pi, 20 * np.log10(abs(h)), 'b')
#     plt.title('Lowpass Filter Frequency Response')
#     plt.ylabel('Amplitude [dB]')
#     plt.grid()

#     # Plot the phase response
#     plt.subplot(2, 1, 2)
#     angles = np.unwrap(np.angle(h))
#     plt.plot(w / np.pi, angles, 'g')
#     plt.ylabel('Angle (radians)')
#     plt.xlabel('Frequency [π*rad/sample]')
#     plt.grid()
#     plt.show()



import numpy as np
from scipy.signal import remez, freqz
import matplotlib.pyplot as plt

def cfirpm(numtaps, bands, ftype, fs=2):
    # 对 bands 进行正规化，使其在 [0, fs/2] 范围内
    bands = np.array(bands)
    bands = (bands + 1) / 2  # 将 [-1, 1] 映射到 [0, 1]

    # 根据给定的带设置 desired amplitude 和 weights
    # MATLAB 中给出了三个带（一个通带和两个阻带）
    # 我们需要在 Python 中对应设置它们
    # 但由于 Python 中的 remez 需要带的 '开始' 和 '结束'，我们有 4 个带边界
    # 而且 'desired' 数组需要对每个频带提供一个目标幅度值
    desired = [0, 1, 0]  # Passband, Stopband, Passband
    weight = [1, 1, 1]

    # 使用 remez 计算滤波器系数
    # numtaps+1 是因为 remez 设计长度为 numtaps+1 的滤波器
    taps = remez(numtaps+1, bands, desired, weight, Hz=fs)
    return taps

# 设计滤波器
b = cfirpm(30, [-1, -0.5, -0.4, 0.7, 0.8, 1], 'lowpass')

# 计算频率响应
w, h = freqz(b, worN=8192, whole=True)

# 绘制幅度响应
plt.subplot(2, 1, 1)
plt.plot(w / np.pi, 20 * np.log10(abs(h)), 'b')
plt.title('Lowpass Filter Frequency Response')
plt.ylabel('Amplitude [dB]')
plt.grid(True)

# 绘制相位响应
plt.subplot(2, 1, 2)
plt.plot(w / np.pi, np.unwrap(np.angle(h)) * 180 / np.pi, 'g')
plt.ylabel('Phase (degrees)')
plt.xlabel('Normalized Frequency (x π rad/sample)')
plt.grid(True)

plt.tight_layout()
plt.show()