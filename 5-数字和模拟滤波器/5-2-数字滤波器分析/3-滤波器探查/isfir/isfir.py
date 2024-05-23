import numpy as np

def is_fir(b, a):  
    return np.allclose(a, [1.0] + [0.0] * (len(a) - 1))   