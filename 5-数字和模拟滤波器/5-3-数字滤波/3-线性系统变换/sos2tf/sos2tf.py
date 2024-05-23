from scipy.signal import sos2tf as scipy_sos2tf

def sos2tf(sos, gain=1):

    b, a = scipy_sos2tf(sos)
    b *= gain
    return b, a
