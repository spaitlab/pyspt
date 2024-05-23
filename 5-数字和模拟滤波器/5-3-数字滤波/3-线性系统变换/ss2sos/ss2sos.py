from scipy.signal import ss2tf, tf2sos

def ss2sos(A, B, C, D):
    
    b, a = ss2tf(A, B, C, D)
    sos = tf2sos(b, a)
    
    return sos