from scipy.signal import sos2zpk

def sos2zp(sos, gain=1):
    # 使用原始的sos2zpk函数转换为零极点表示形式
    Z, P, K = sos2zpk(sos)
    # 得到零极点形式下增益
    K *= gain
    
    return Z, P, K