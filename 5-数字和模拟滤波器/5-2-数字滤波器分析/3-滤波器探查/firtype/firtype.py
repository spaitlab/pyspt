import numpy as np 

def firtype(b):
    N = len(b)
    
    # 序列长度为偶数，则为奇阶(序列长度=阶数+1)
    if N % 2 == 0:
        if np.allclose(b, b[::-1]):
            return 2  # 奇阶对称，type2
        elif np.allclose(b, -b[::-1]):
            return 4  # 奇阶反对称，type4
        return False 
    
    elif N % 2 != 0:
    # 序列长度为奇数，则为偶阶
        middle_index = N // 2   
        if np.allclose(b[:middle_index], b[middle_index + 1:][::-1]):  # 偶阶对称，type1
            return 1 
        elif np.allclose(b[:middle_index], -b[middle_index + 1:][::-1]):  # 偶阶反对称，type3
            return 3  
        else:  
            return False
        
    else:  
        return False   
     