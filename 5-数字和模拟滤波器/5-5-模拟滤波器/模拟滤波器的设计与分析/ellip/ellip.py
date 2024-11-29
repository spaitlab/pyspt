import numpy as np
from scipy import signal

def ellip(N, rp, rs, Wn, btype='low', analog=False, output='ba'):
    return signal.ellip(N, rp, rs, Wn, btype, analog, output)