import numpy as np
from scipy import signal

def cheby1_filter(order, rp, Wn, btype='low', analog=False, fs=None):
    b, a = signal.cheby1(order, rp, Wn, btype=btype, analog=analog, fs=fs)
    return b, a