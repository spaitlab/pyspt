import numpy as np

def ismaxphase(b, a, tol=np.finfo(float).eps**(2/3)):
    # 判断是否为最大相位滤波器
    if isinstance(b, np.ndarray) and isinstance(a, np.ndarray):
        roots_a = np.roots(a)
        roots_b = np.roots(b)
        if np.all(np.real(roots_a) < 0) and np.all(np.abs(roots_b) < 1 + tol):
            return True
        else:
            return False
    elif isinstance(b, np.ndarray) and a is None:
        sos = b
        if sos.shape[1] != 6 or sos.shape[0] < 2:
            raise ValueError("Invalid sos matrix shape")
        
        for i in range(sos.shape[0]):
            bi = sos[i, :3]
            ai = sos[i, 3:]
            if not ismaxphase(bi, ai, tol):
                return False
        return True
    elif b is not None and a is None:
        d = b
        b, a = d.num, d.den
        return ismaxphase(b, a, tol)
    else:
        raise ValueError("Invalid input")
