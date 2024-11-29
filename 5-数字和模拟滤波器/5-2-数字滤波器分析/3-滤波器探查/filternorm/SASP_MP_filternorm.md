# 信号处理仿真与应用 - 数字和模拟滤波器  - 数字滤波器分析

## MATLAB函数描述：filternorm 

函数来源：[MATLAB filternorm](https://ww2.mathworks.cn/help/signal/ref/filternorm.html)

### 语法

L = filternorm(b,a)
L = filternorm(b,a,pnorm)
L = filternorm(b,a,2,tol)

### 说明

滤波器规范的典型用途是在数字滤波器缩放中以减少量化效果。缩放通常可以提高滤波器的信噪比，而不会导致数据溢出。您也可以使用2-范数来计算滤波器的脉冲响应的能量。
L＝滤波器范数（b，a）计算由b中的分子系数和a中的分母系数定义的数字滤波器的2-范数。
L=滤波器范数（b，a，pnorm）计算数字滤波器的2-或无穷远范数（inf范数），其中pnorm是2或inf。
L=滤波器范数（b，a，2，tol）计算具有指定容差tol的IIR滤波器的2-范数。只能为IIR 2-范数计算指定公差。在这种情况下pnorm必须是2。如果未指定tol，则默认为10<sup>*-8*</sup>。

### 输入参数

b,a  — 传递函数系数
向量 
传递函数系数，指定为向量。将传递函数以a,b的形式表达出来，传递函数形式如下：
$$
H(z)=\frac{B(Z)}{A(Z)}=\frac{b_{1}+b_{2}z^{-1}...+b_{n+1}z^{-n}}{a_{1}+a_{2}z^{-1}...+a_{m+1}z^{-m}}
$$
示例：指定归一化 3 dB 频率为 0.5π rad/sample的三阶巴特沃斯滤波器 。`b = [1 3 3 1]/6``a = [3 0 1 0]/3` 

数据类型: single | double

复数支持：是的



tol — IIR滤波器有效脉冲响应长度公差
1e-8（默认）|正标量                                                                                                                                                        IIR滤波器有效脉冲响应长度的容差，指定为正数。公差决定了绝对可和序列中的项，在该序列之后，后续项被认为是0。默认公差为1e-8。增加容差将返回更短的有效脉冲响应序列长度。减小公差将返回更长的有效脉冲响应序列长度。

pnorm — 范数量
2 or inf                                                                                                                                                                                     范数量，指定计算2-范数或是无穷范数

### 输出参量

L — 范数值
double
范数值，作为数字返回。



## Python函数描述：filternorm

函数来源：自定义

### 滤波器阶跃响应函数定义：

```
import numpy as np
from scipy.signal import freqz, butter,firwin
from scipy.linalg import norm
from impz import impz
from impzlength import impzlength

def filternorm(b, a, pnorm):
    # Check number of input arguments
    if not (2 <= len(locals()) <= 4):
        raise ValueError("Wrong number of input arguments. Expected 2 to 4.")

    # Set pnorm to 2 if only two input arguments are provided
    if len(locals()) == 2:
        pnorm = 2

    # Validate input data types and values
    if not isinstance(b, np.ndarray) or not isinstance(a, np.ndarray):
        raise TypeError("Inputs 'b' and 'a' must be numpy arrays.")
    if not np.issubdtype(b.dtype, np.float64) or not np.issubdtype(a.dtype, np.float64):
        raise TypeError("Inputs 'b' and 'a' must be of data type float64.")

    # Validate pnorm
    pnormScalar = pnorm
    if not (np.isinf(pnormScalar) or pnormScalar == 2):
        raise ValueError("Invalid value for pnorm. Must be either Inf or 2.")

    # Convert b and a to row vectors
    bRow = np.squeeze(b)
    aRow = a

    # Check for zero-leading denominator coefficient
    # if aRow[0] == 0:
    #     raise ValueError("Zero-leading denominator coefficient is not allowed.")

    # Check stability and FIR/IIR
    isStable, isFIR = isstableTF(bRow, aRow)
    if not isStable:
        raise ValueError("Filter is unstable.")

    if np.isinf(pnormScalar):
        # Compute inf-norm (maximum magnitude of frequency response)
        _, h = freqz(bRow, aRow, worN=1024)
        s = np.max(np.abs(h))
    else:  # pnormScalar == 2
        if isFIR:
            # Compute 2-norm for FIR filter
            s = norm(bRow, ord=-2)
        else:
            # Compute 2-norm for IIR filter
            tol = 1e-8  # Default tolerance for IIR filter
            maxradius = np.max(np.abs(np.roots(aRow)))
            # Check stability using maximum radius of poles
            if maxradius >= 1:
                raise ValueError("Filter is unstable (numerical precision issue).")

            # Determine length of impulse response
            N = impzlength(bRow, aRow, tol)
            H = impz(bRow, aRow, N)
            s = norm(H, ord=-2)

    return s

def isstableTF(num, den):
    # Determine if filter is FIR or IIR
    firFlag = np.isscalar(den) or len(den) == 1
    if firFlag:
        # FIR filter is always stable
        return True, True
    else:
        # Check stability of IIR filter
        return np.all(np.abs(np.roots(den)) < 1), False
```


该函数用于计算数字滤波器的2-范数与无穷范数，这可以用于评估滤波器的性能。

### 参数
- `b`: 滤波器的分子系数，可以是传递函数系数向量。
- `a`: 滤波器的分母系数，可以是传递函数系数向量或SOS（second-order sections）矩阵。
- `pnorm`: 范数类型，可选参数。如果未指定，默认为2（即2-范数）。

### 返回值
- 滤波器的范数`s`，作为浮点数返回。

### 注意事项
- 确保`b`和`a`参数为NumPy数组或可转换为NumPy数组的序列类型。
- `b`和`a`的长度应该匹配。
- `pnorm`参数应该为2（2-范数）或无穷大（无穷范数）。

### 函数工作原理
1. 函数首先检查`b`和`a`参数的类型和长度。
2. 函数计算滤波器系数`b`和`a`的范数，根据`pnorm`参数的值确定使用2-范数还是无穷范数。
3. 对于无穷范数，函数使用`scipy.signal.freqz`计算频率响应，并取其最大绝对值。
4. 对于2-范数，函数计算滤波器的脉冲响应，并取其2-范数。

### 使用场景
- 设计和测试数字滤波器时，用于评估滤波器的性能。
- 在信号处理领域，比较不同滤波器的性能。

### 改进建议
- 函数内部的一些辅助函数（如`impz`、`impzlength`）应该确保被正确导入和可用。
- 可以考虑为`pnorm`参数提供一个合理的默认值，以简化函数调用。
- 对于复杂的滤波器类型（如SOS），可以增加更多的错误检查，以确保输入数据的正确性。
- 函数可以提供更详细的文档字符串，说明每个参数的作用和预期类型，以及函数的返回值。
- 可以增加测试用例，以验证函数在不同滤波器配置下的正确性和鲁棒性。

## Prompt 1 ： 生成 Python filternorm 函数

参考下面MATLAB代码的filternorm函数
```
narginchk(2,4);

if nargin == 2, pnorm = 2; end

% Check the input data type. Single precision is not supported.
validateattributes(b,{'double'},{'nonsparse','nonempty'},mfilename,'B',1);
validateattributes(a,{'double'},{'nonsparse','nonempty'},mfilename,'A',2);
validateattributes(pnorm,{'double'},{},mfilename,'PNORM',3);

pnormScalar = pnorm(1);
coder.internal.assert((pnormScalar == Inf) || (pnormScalar == 2),...
    'signal:filternorm:InvalidPnorm');

bRow = b(:).';
aRow = a(:).';

coder.internal.errorIf(aRow(1) == 0,'signal:filternorm:ZeroLeadDenCoeff');

% Check for stability and FIR/IIR
[isStable,isFIR] = isstableTF(bRow,aRow);
coder.internal.assert(isStable,'signal:filternorm:SignalErrUnstable');

if isinf(pnormScalar)
    % inf-norm is simply given by the max of the magnitude response
    h = freqz(bRow,aRow,1024);
    s = max(abs(h));
else
    % For pnorm = 2
    if isFIR
        % For a FIR filter, compute 2-norm by simply summing up the square
        % of the impulse response.
        s = norm(bRow,pnormScalar);
        if coder.target('MATLAB') && nargin == 4
            warning(message('signal:filternorm:Ignore'));
        end
    else
        % Default tolerance
        if nargin < 4, tol = 1e-8; end

        % For an IIR filter, compute 2-norm by approximating the impulse
        % response as finite, alternatively use residues to compute contour
        % integral
        maxradius = max(abs(roots(aRow)));
        % Include an extra check for stability in case numerical roundoff
        % problems
        coder.internal.errorIf(maxradius >= 1,'signal:filternorm:SignalErrUnstableNumPrec');

        % Determine the number of impulse response points
        N = impzlength(bRow,aRow,tol);
        H = impz(bRow,aRow,N);
        s = norm(H,pnormScalar);
    end
end

end

function [stableFlag,firFlag] = isstableTF(num,den)

firFlag = signal.internal.isfir(num,den);
if firFlag
    % Section is FIR, always stable
    stableFlag = true;
else
    stableFlag = signal.internal.isstable(den);
end

end
```

这个函数的功能是生成滤波器的2-范数或无穷范数，输入参数为滤波器的b，a值。请根据这个函数完成Python语言filternorm函数的编写，便于类似应用调用。

### LLM 生成 ： Python filternorm 函数

这是该函数的定义：

    import numpy as np
    from scipy.signal import freqz, butter,firwin
    from scipy.linalg import norm
    from impz import impz
    from impzlength import impzlength
    
    def filternorm(b, a, pnorm):
        # Check number of input arguments
        if not (2 <= len(locals()) <= 4):
            raise ValueError("Wrong number of input arguments. Expected 2 to 4.")
    
        # Set pnorm to 2 if only two input arguments are provided
        if len(locals()) == 2:
            pnorm = 2
    
        # Validate input data types and values
        if not isinstance(b, np.ndarray) or not isinstance(a, np.ndarray):
            raise TypeError("Inputs 'b' and 'a' must be numpy arrays.")
        if not np.issubdtype(b.dtype, np.float64) or not np.issubdtype(a.dtype, np.float64):
            raise TypeError("Inputs 'b' and 'a' must be of data type float64.")
    
        # Validate pnorm
        pnormScalar = pnorm
        if not (np.isinf(pnormScalar) or pnormScalar == 2):
            raise ValueError("Invalid value for pnorm. Must be either Inf or 2.")
    
        # Convert b and a to row vectors
        bRow = np.squeeze(b)
        aRow = a
    
        # Check for zero-leading denominator coefficient
        # if aRow[0] == 0:
        #     raise ValueError("Zero-leading denominator coefficient is not allowed.")
    
        # Check stability and FIR/IIR
        isStable, isFIR = isstableTF(bRow, aRow)
        if not isStable:
            raise ValueError("Filter is unstable.")
    
        if np.isinf(pnormScalar):
            # Compute inf-norm (maximum magnitude of frequency response)
            _, h = freqz(bRow, aRow, worN=1024)
            s = np.max(np.abs(h))
        else:  # pnormScalar == 2
            if isFIR:
                # Compute 2-norm for FIR filter
                s = norm(bRow, ord=-2)
            else:
                # Compute 2-norm for IIR filter
                tol = 1e-8  # Default tolerance for IIR filter
                maxradius = np.max(np.abs(np.roots(aRow)))
                # Check stability using maximum radius of poles
                if maxradius >= 1:
                    raise ValueError("Filter is unstable (numerical precision issue).")
    
                # Determine length of impulse response
                N = impzlength(bRow, aRow, tol)
                H = impz(bRow, aRow, N)
                s = norm(H, ord=-2)
    
        return s
    
    def isstableTF(num, den):
        # Determine if filter is FIR or IIR
        firFlag = np.isscalar(den) or len(den) == 1
        if firFlag:
            # FIR filter is always stable
            return True, True
        else:
            # Check stability of IIR filter
            return np.all(np.abs(np.roots(den)) < 1), False


## Prompt 2 ： 应用生成的 Python filternorm.py函数做MATLAB示例对应

这是MATLAB程序1，要求计算容差为10<sup>*-10*</sup>的巴特沃斯IIR滤波器的2-范数。指定0.5πrad/s的归一化截止频率和5阶滤波器。

```
[b,a] = butter(5,0.5);
L2 = filternorm(b,a,2,1e-10)
```

这是filternorm函数定义

    import numpy as np
    from scipy.signal import freqz, butter,firwin
    from scipy.linalg import norm
    from impz import impz
    from impzlength import impzlength
    
    def filternorm(b, a, pnorm):
        # Check number of input arguments
        if not (2 <= len(locals()) <= 4):
            raise ValueError("Wrong number of input arguments. Expected 2 to 4.")
    
        # Set pnorm to 2 if only two input arguments are provided
        if len(locals()) == 2:
            pnorm = 2
    
        # Validate input data types and values
        if not isinstance(b, np.ndarray) or not isinstance(a, np.ndarray):
            raise TypeError("Inputs 'b' and 'a' must be numpy arrays.")
        if not np.issubdtype(b.dtype, np.float64) or not np.issubdtype(a.dtype, np.float64):
            raise TypeError("Inputs 'b' and 'a' must be of data type float64.")
    
        # Validate pnorm
        pnormScalar = pnorm
        if not (np.isinf(pnormScalar) or pnormScalar == 2):
            raise ValueError("Invalid value for pnorm. Must be either Inf or 2.")
    
        # Convert b and a to row vectors
        bRow = np.squeeze(b)
        aRow = a
    
        # Check for zero-leading denominator coefficient
        # if aRow[0] == 0:
        #     raise ValueError("Zero-leading denominator coefficient is not allowed.")
    
        # Check stability and FIR/IIR
        isStable, isFIR = isstableTF(bRow, aRow)
        if not isStable:
            raise ValueError("Filter is unstable.")
    
        if np.isinf(pnormScalar):
            # Compute inf-norm (maximum magnitude of frequency response)
            _, h = freqz(bRow, aRow, worN=1024)
            s = np.max(np.abs(h))
        else:  # pnormScalar == 2
            if isFIR:
                # Compute 2-norm for FIR filter
                s = norm(bRow, ord=-2)
            else:
                # Compute 2-norm for IIR filter
                tol = 1e-8  # Default tolerance for IIR filter
                maxradius = np.max(np.abs(np.roots(aRow)))
                # Check stability using maximum radius of poles
                if maxradius >= 1:
                    raise ValueError("Filter is unstable (numerical precision issue).")
    
                # Determine length of impulse response
                N = impzlength(bRow, aRow, tol)
                H = impz(bRow, aRow, N)
                s = norm(H, ord=-2)
    
        return s
    
    def isstableTF(num, den):
        # Determine if filter is FIR or IIR
        firFlag = np.isscalar(den) or len(den) == 1
        if firFlag:
            # FIR filter is always stable
            return True, True
        else:
            # Check stability of IIR filter
            return np.all(np.abs(np.roots(den)) < 1), False

请应用自定义filternorm函数，实现上述MATLAB程序的功能，并给出Python程序。

### LLM 生成 ： 
```
import numpy as np
from scipy.signal import freqz, butter,firwin
from filternorm import filternorm

# 示例1
# 设计 Butterworth 低通滤波器
order = 5        # 阶数
cutoff = 0.5     # 截止频率
# 设计低通 Butterworth 滤波器
b,a  = butter(order, cutoff, btype='low')
pnorm = 2                      # P-norm (default: 2)
norm_value = filternorm(b, a, pnorm)
print(f"Norm value: {norm_value}")
```

## Prompt 3 ： 应用生成的 Python filternorm.py 函数做MATLAB示例对应

这是MATLAB程序2，要求计算30阶FIR Hilbert变换器的无穷远范数，其归一化跃迁宽度0.2πrad/s。参考下面Matlab代码，给出Python实现代码：

**Matlab代码**

```
b = firpm(30,[.1 .9],[1 1],'Hilbert');
Linf = filternorm(b,1,inf)
```

这是Python filternorm.py
```
import numpy as np
from scipy.signal import freqz, butter,firwin
from scipy.linalg import norm
from impz import impz
from impzlength import impzlength

def filternorm(b, a, pnorm):
    # Check number of input arguments
    if not (2 <= len(locals()) <= 4):
        raise ValueError("Wrong number of input arguments. Expected 2 to 4.")

    # Set pnorm to 2 if only two input arguments are provided
    if len(locals()) == 2:
        pnorm = 2

    # Validate input data types and values
    if not isinstance(b, np.ndarray) or not isinstance(a, np.ndarray):
        raise TypeError("Inputs 'b' and 'a' must be numpy arrays.")
    if not np.issubdtype(b.dtype, np.float64) or not np.issubdtype(a.dtype, np.float64):
        raise TypeError("Inputs 'b' and 'a' must be of data type float64.")

    # Validate pnorm
    pnormScalar = pnorm
    if not (np.isinf(pnormScalar) or pnormScalar == 2):
        raise ValueError("Invalid value for pnorm. Must be either Inf or 2.")

    # Convert b and a to row vectors
    bRow = np.squeeze(b)
    aRow = a

    # Check for zero-leading denominator coefficient
    # if aRow[0] == 0:
    #     raise ValueError("Zero-leading denominator coefficient is not allowed.")

    # Check stability and FIR/IIR
    isStable, isFIR = isstableTF(bRow, aRow)
    if not isStable:
        raise ValueError("Filter is unstable.")

    if np.isinf(pnormScalar):
        # Compute inf-norm (maximum magnitude of frequency response)
        _, h = freqz(bRow, aRow, worN=1024)
        s = np.max(np.abs(h))
    else:  # pnormScalar == 2
        if isFIR:
            # Compute 2-norm for FIR filter
            s = norm(bRow, ord=-2)
        else:
            # Compute 2-norm for IIR filter
            tol = 1e-8  # Default tolerance for IIR filter
            maxradius = np.max(np.abs(np.roots(aRow)))
            # Check stability using maximum radius of poles
            if maxradius >= 1:
                raise ValueError("Filter is unstable (numerical precision issue).")

            # Determine length of impulse response
            N = impzlength(bRow, aRow, tol)
            H = impz(bRow, aRow, N)
            s = norm(H, ord=-2)

    return s

def isstableTF(num, den):
    # Determine if filter is FIR or IIR
    firFlag = np.isscalar(den) or len(den) == 1
    if firFlag:
        # FIR filter is always stable
        return True, True
    else:
        # Check stability of IIR filter
        return np.all(np.abs(np.roots(den)) < 1), False
```


### LLM 生成 ： 

下面这是调用程序
```
import numpy as np
from scipy.signal import freqz, butter,firwin
from filternorm import filternorm


# 示例2
# 设计希尔伯特滤波器参数
numtaps = 30  # 滤波器阶数
bands = [0.1, 0.9]  # 频带边界
desired = [1, 1]  # 理想增益

# 设计希尔伯特滤波器
b = firwin(numtaps, cutoff=bands, pass_zero=False, fs=3.0)
a = np.array([1.00000000e+00])
#a = np.array([1, -0.8, 0.3])
norm_value = filternorm(b, a, np.inf)
print(f"Norm value: {norm_value}")
```



