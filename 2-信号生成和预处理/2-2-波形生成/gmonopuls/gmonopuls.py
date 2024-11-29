"""
This file contains code used in "GPT-PySPT", by Zhiguo Zhou, available from SpaitLab
Copyright 2024 Zhiguo Zhou
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""
import numpy as np

def gmonopuls(t, fc):
    tau = 1 / (2 * np.pi * fc)
    a = 1 / (2 * tau**2)
    y = t * np.exp(-a * t**2)
    peak_value = np.max(y)
    y = y / (peak_value+1e-15)
    return y



