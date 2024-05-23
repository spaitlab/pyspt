# 信号处理仿真与应用 - 数字和模拟滤波器  - 数字滤波器分析

## MATLAB函数描述：filtord

函数来源：[MATLAB filtord ](https://ww2.mathworks.cn/help/signal/ref/filtord.html)

### 语法

n = filtord(b,a)
n = filtord(sos)
n = filtord(d)

### 说明

n=filtord(b，a)返回由分子系数b和分母系数a指定的因果有理系统函数的滤波器阶数n。
n=filtord（sos）返回由二阶截面矩阵sos指定的滤波器的滤波器阶数。sos是一个K-by-6矩阵。节数K必须大于或等于2。每一行sos对应于二阶滤波器的系数。二阶截面矩阵的第i行对应于[bi（1）bi（2）bi（3）ai（1）ai（2）ai（3）]。                                                                                                                                                                                n=filtord（d）返回数字滤波器d的滤波器阶数n。使用函数designfilt生成d。                                                                                           

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



sos — 二阶截面系数
矩阵
二阶截面系数，指定为矩阵。 是一个 *K*×6 矩阵，其中截面数 *K* 必须大于或等于2。如果节数小 2，该函数将输入视为分子向量。每行对应于二阶的系数滤波器。第 *i*行对应于 `sos``[bi(1) bi(2) bi(3) ai(1) ai(2) ai(3)]`

*第K*个二阶的系统函数 filter 具有有理 Z 变换
$$
H(z)=\frac{B_{k}(1)+B_{k}(2)z^{-1}+B_{k}(3)z^{-2}}{A_{k}(1)+A_{k}(2)z^{-1}+A_{k}(3)z^{-2}}
$$
滤波器的频率响应是在单位圆上评估的系统函数z=e<sup>*j2pif*</sup>

示例：指定三阶巴特沃斯 滤波器，归一化 3 dB 频率 0.5π rad/样本。`s = [2 4 2 6 0 2;3 3 0 6 0 0]`	  复数支持：是`double``single`



d — 数字滤波器
digitalFilter对象数字筛选器，指定为 [`digitalFilter`](https://ww2.mathworks.cn/help/signal/ref/digitalfilter.html) 对象。使用 [`designfilt`](https://ww2.mathworks.cn/help/signal/ref/designfilt.html) 生成数字滤波器 基于频率响应规范。                                                                                                                                                                                     示例：指定归一化 3 dB 频率为 0.5π 的三阶巴特沃斯滤波器 rad/sample。`d = designfilt('lowpassiir','FilterOrder',3,'HalfPowerFrequency',0.5)`



### 输出参量

n —滤波器阶数
整数
滤波器阶数，指定为整数



## Python函数描述：filtord

函数来源：自定义

### 滤波器阶跃响应函数定义：

```
import numpy as np
from scipy.signal import firwin, iirfilter

def filtord(b, *args):
    if len(args) == 0:
        a = [1]  # Assume FIR for now
    else:
        a = args[0]
    a = np.asarray(a)
    
    # Cast to precision rules
    if np.issubdtype(b.dtype, np.float32) or np.issubdtype(a.dtype, np.float32):
        convClass = np.float32
    else:
        convClass = np.float64
    
    b = b.astype(convClass)
    a = a.astype(convClass)

    # If b is SOS or vector
    if len(args) == 0:
        a1 = a
        if np.ndim(b) == 1:
            # If input is column vector, transpose to obtain row vectors
            if b.ndim == 2 and b.shape[1] == 6:
                # Check if input is a valid SOS matrix
                raise ValueError('Invalid SOS matrix dimensions')
            b1 = b.reshape(1, -1)
        else:
            # If input is a matrix, check if it is a valid SOS matrix
            if b.shape[1] != 6:
                raise ValueError('Invalid SOS matrix dimensions')
            b1, a1 = sos2tf(b)
    else:  # If b and a are vectors
        # If b is not a vector, then only one input is supported
        if b.ndim > 1 and b.shape[0] > 1 and b.shape[1] > 1:
            raise ValueError('Invalid number of inputs')
        
        # If a is not a vector
        if a.ndim > 1 and a.shape[0] > 1 and a.shape[1] > 1:
            raise ValueError('Input not supported')
        
        b1 = b
        a1 = a
        
        
        # If input is column vector, transpose to obtain row vectors
        if b.ndim == 2 and b.shape[1] == 1:
            b1 = b.T
        
        if a.ndim == 2 and a.shape[1] == 1:
            a1 = a.T

    # Normalizing the filter coefficients
    if not np.allclose(b1, 0):
        maxCoefNum = np.max(np.abs(b1))
        b1 /= maxCoefNum
    
    if not np.allclose(a1, 0):
        maxCoefDen = np.max(np.abs(a1))
        a1 /= maxCoefDen
    
    # Returning the index of the last nonzero coefficient
    nZeroLastNum = np.flatnonzero(b1)[-1] if not np.allclose(b1, 0) else 0
    nZeroLastDen = np.flatnonzero(a1)[-1] if not np.allclose(a1, 0) else 0
    
    # Filter order n is maximum of the last nonzero coefficient subtracted by 1
    n = max(nZeroLastNum, nZeroLastDen)
    
    return n


# Placeholder for sos2tf function (not implemented in this example)
def sos2tf(b):
    # Implement SOS to TF conversion if needed
    raise NotImplementedError("sos2tf function is not implemented")
```


该函数用于确定数字滤波器的阶数，即滤波器中系数不为零的项数。

### 参数
- `b`: 滤波器的分子系数，可以是传递函数系数向量或SOS（second-order sections）矩阵。
- `*args`: 可变参数，用于指定滤波器的分母系数`a`（可选）。

### 返回值
- 滤波器的阶数`n`，作为整数返回。

### 注意事项
- 确保`b`和`a`参数为NumPy数组或可转换为NumPy数组的序列类型。
- 当`b`是SOS矩阵时，确保其形状正确，即每一行有6个元素。
- 如果指定了`a`，`b`和`a`的长度应该匹配。

### 函数工作原理
1. 函数首先检查`b`是否为SOS矩阵或传递函数系数向量。
2. 函数计算滤波器系数`b`和`a`的阶数，即最后一项的索引。
3. 函数返回滤波器的阶数，即最大索引减一。

### 使用场景
- 设计和测试数字滤波器时，用于获取滤波器的阶数，以便了解滤波器的复杂度和性能。
- 在信号处理领域，分析不同滤波器阶数对系统性能的影响。

### 改进建议
- 可以考虑为`a`参数提供一个合理的默认值，以简化函数调用。
- 对于复杂的滤波器类型（如SOS），可以增加更多的错误检查，以确保输入数据的正确性。
- 函数可以提供更详细的文档字符串，说明每个参数的作用和预期类型，以及函数的返回值。
- 可以增加测试用例，以验证函数在不同滤波器配置下的正确性和鲁棒性。
- 考虑将一些复杂的操作（如SOS到TF的转换）分离出来，以使函数更加灵活和易于维护。

## Prompt 1 ： 生成 Python filtord 函数

参考下面MATLAB代码filtord函数
```
narginchk(1,2);

if coder.target('MATLAB')
    % MATLAB
    n = efiltord(b,varargin{:});
else
    % Code generation  
    if nargin == 1
        allConst = coder.internal.isConst(b);
    else
        allConst = coder.internal.isConst(b) && coder.internal.isConst(varargin{1});
    end
    
    if allConst && coder.internal.isCompiled
        % Constant Inputs
        n = coder.const(@feval,'filtord',b,varargin{:});        
    else
        % Variable Inputs
        n = efiltord(b,varargin{:});
    end    
end

end

function n = efiltord(b,varargin)

if nargin == 1
    a = 1; % Assume FIR for now
else
    a = varargin{1};
end

validateattributes(b,{'double','single'},{'2d'},'filtord');
validateattributes(a,{'double','single'},{'2d'},'filtord');

% Cast to precision rules
% Single/double datatype check and conversion
if isa(b,'single') || isa(a,'single')
    convClass = 'single';
else
    convClass = 'double';
end

b = cast(b,convClass);
a = cast(a,convClass);

coder.varsize('b1','a1');

% If b is SOS or vector,
if nargin == 1
    a1 = a;
    if isvector(b)
        % If input is column vector transpose to obtain row vectors
        if iscolumn(b)
            b1 = b.';
        else
            b1 = b;
        end
    else
        % If input is a matrix, check if it is a valid SOS matrix
        coder.internal.errorIf(size(b,2) ~= 6,'signal:signalanalysisbase:invalidinputsosmatrix');
         
        % Get transfer function
        [b1,a1] = sos2tf(b);
    end
    
else    % If b and a are vectors
    
    % If b is not a vector, then only one input is supported
    coder.internal.errorIf(size(b,1)>1 && size(b,2)>1,'signal:signalanalysisbase:invalidNumInputs');
    
    % If a is not a vector
    coder.internal.errorIf(size(a,1)>1 && size(a,2)>1,'signal:signalanalysisbase:inputnotsupported');
    
    
    b1 = b;
    a1 = a;
    
    % If input is column vector transpose to obtain row vectors
    if iscolumn(b)
        b1 = b.';
    end
    
    if iscolumn(a)
        a1 = a.';
    end
end

% Normalizing the filter coefficients
if ~isempty(b1)
    maxCoefNum = max(abs(b1),[],2);
    if maxCoefNum ~= 0
        b1 = b1/maxCoefNum(1);
    end
end

if ~isempty(a1)
    maxCoefDen = max(abs(a1),[],2);
    if maxCoefDen ~= 0
        a1 = a1/maxCoefDen(1);
    end
end

nZeroLastNum = 0;
nZeroLastDen = 0;

% Returning the index of the last nonzero coefficient
if ~isempty(b1)
    nZeroLastDen = find(b1(:)~=0, 1, 'last');
end

if ~isempty(a1)
    nZeroLastNum = find(a1(:)~=0, 1, 'last');
end

if isempty(nZeroLastDen)
    nZeroLastDen = 0;
end

if isempty(nZeroLastNum)
    nZeroLastNum = 0;
end

% filter order n is maximum of the last nonzero coefficient subtracted by 1
n = max(nZeroLastNum(1),nZeroLastDen(1)) - 1;

end
```

这个函数的功能是生成滤波器的阶数，输入参数为滤波器的b，a值。请根据这个函数完成Python语言filtord函数的编写，便于类似应用调用。

### LLM 生成 ： Python filtord 函数

这是该函数的定义：

    import numpy as np
    from scipy.signal import firwin, iirfilter
    
    def filtord(b, *args):
        if len(args) == 0:
            a = [1]  # Assume FIR for now
        else:
            a = args[0]
        a = np.asarray(a)
        
        # Cast to precision rules
        if np.issubdtype(b.dtype, np.float32) or np.issubdtype(a.dtype, np.float32):
            convClass = np.float32
        else:
            convClass = np.float64
        
        b = b.astype(convClass)
        a = a.astype(convClass)
    
        # If b is SOS or vector
        if len(args) == 0:
            a1 = a
            if np.ndim(b) == 1:
                # If input is column vector, transpose to obtain row vectors
                if b.ndim == 2 and b.shape[1] == 6:
                    # Check if input is a valid SOS matrix
                    raise ValueError('Invalid SOS matrix dimensions')
                b1 = b.reshape(1, -1)
            else:
                # If input is a matrix, check if it is a valid SOS matrix
                if b.shape[1] != 6:
                    raise ValueError('Invalid SOS matrix dimensions')
                b1, a1 = sos2tf(b)
        else:  # If b and a are vectors
            # If b is not a vector, then only one input is supported
            if b.ndim > 1 and b.shape[0] > 1 and b.shape[1] > 1:
                raise ValueError('Invalid number of inputs')
            
            # If a is not a vector
            if a.ndim > 1 and a.shape[0] > 1 and a.shape[1] > 1:
                raise ValueError('Input not supported')
            
            b1 = b
            a1 = a
            
            
            # If input is column vector, transpose to obtain row vectors
            if b.ndim == 2 and b.shape[1] == 1:
                b1 = b.T
            
            if a.ndim == 2 and a.shape[1] == 1:
                a1 = a.T
    
        # Normalizing the filter coefficients
        if not np.allclose(b1, 0):
            maxCoefNum = np.max(np.abs(b1))
            b1 /= maxCoefNum
        
        if not np.allclose(a1, 0):
            maxCoefDen = np.max(np.abs(a1))
            a1 /= maxCoefDen
        
        # Returning the index of the last nonzero coefficient
        nZeroLastNum = np.flatnonzero(b1)[-1] if not np.allclose(b1, 0) else 0
        nZeroLastDen = np.flatnonzero(a1)[-1] if not np.allclose(a1, 0) else 0
        
        # Filter order n is maximum of the last nonzero coefficient subtracted by 1
        n = max(nZeroLastNum, nZeroLastDen)
        
        return n
    
    
    # Placeholder for sos2tf function (not implemented in this example)
    def sos2tf(b):
        # Implement SOS to TF conversion if needed
        raise NotImplementedError("sos2tf function is not implemented")


## Prompt 2 ： 应用生成的 Python filtord.py函数做MATLAB示例对应

这是MATLAB程序1，要求采用窗口法设计了一个归一化截止频率为0.5πrad/sample的20阶FIR滤波器。验证滤波器阶数。

```
b = fir1(20,0.5);
n = filtord(b)
```

这是filtord函数定义

    import numpy as np
    from scipy.signal import firwin, iirfilter
    
    def filtord(b, *args):
        if len(args) == 0:
            a = [1]  # Assume FIR for now
        else:
            a = args[0]
        a = np.asarray(a)
        
        # Cast to precision rules
        if np.issubdtype(b.dtype, np.float32) or np.issubdtype(a.dtype, np.float32):
            convClass = np.float32
        else:
            convClass = np.float64
        
        b = b.astype(convClass)
        a = a.astype(convClass)
    
        # If b is SOS or vector
        if len(args) == 0:
            a1 = a
            if np.ndim(b) == 1:
                # If input is column vector, transpose to obtain row vectors
                if b.ndim == 2 and b.shape[1] == 6:
                    # Check if input is a valid SOS matrix
                    raise ValueError('Invalid SOS matrix dimensions')
                b1 = b.reshape(1, -1)
            else:
                # If input is a matrix, check if it is a valid SOS matrix
                if b.shape[1] != 6:
                    raise ValueError('Invalid SOS matrix dimensions')
                b1, a1 = sos2tf(b)
        else:  # If b and a are vectors
            # If b is not a vector, then only one input is supported
            if b.ndim > 1 and b.shape[0] > 1 and b.shape[1] > 1:
                raise ValueError('Invalid number of inputs')
            
            # If a is not a vector
            if a.ndim > 1 and a.shape[0] > 1 and a.shape[1] > 1:
                raise ValueError('Input not supported')
            
            b1 = b
            a1 = a
            
            
            # If input is column vector, transpose to obtain row vectors
            if b.ndim == 2 and b.shape[1] == 1:
                b1 = b.T
            
            if a.ndim == 2 and a.shape[1] == 1:
                a1 = a.T
    
        # Normalizing the filter coefficients
        if not np.allclose(b1, 0):
            maxCoefNum = np.max(np.abs(b1))
            b1 /= maxCoefNum
        
        if not np.allclose(a1, 0):
            maxCoefDen = np.max(np.abs(a1))
            a1 /= maxCoefDen
        
        # Returning the index of the last nonzero coefficient
        nZeroLastNum = np.flatnonzero(b1)[-1] if not np.allclose(b1, 0) else 0
        nZeroLastDen = np.flatnonzero(a1)[-1] if not np.allclose(a1, 0) else 0
        
        # Filter order n is maximum of the last nonzero coefficient subtracted by 1
        n = max(nZeroLastNum, nZeroLastDen)
        
        return n
    
    
    # Placeholder for sos2tf function (not implemented in this example)
    def sos2tf(b):
        # Implement SOS to TF conversion if needed
        raise NotImplementedError("sos2tf function is not implemented")

请应用自定义filtord函数，实现上述MATLAB程序的功能，并给出Python程序。

### LLM 生成 ： 
```
import numpy as np
from scipy.signal import firwin, iirfilter
from filtord import filtord


# 示例1
# 设计 FIR 滤波器
order = 20  # 滤波器阶数
cutoff_freq = 0.5  # 截止频率

# 计算截止频率对应的归一化频率
nyquist_freq = 1  # Nyquist 频率，即采样频率的一半
normalized_cutoff_freq = cutoff_freq / nyquist_freq

# 使用 firwin 函数设计 FIR 滤波器
b = firwin(order + 1, normalized_cutoff_freq, window='hamming')
n = filtord(b)
print("Filter order (n):", n)
```

## Prompt 3 ： 应用生成的 Python filtord.py 函数做MATLAB示例对应

这是MATLAB程序2，要求根据同一组规范设计FIR等波纹滤波器和IIR巴特沃斯滤波器。确定两种设计之间滤波器顺序的差异。参考下面Matlab代码，给出Python实现代码：

**Matlab代码**

```
fir = designfilt('lowpassfir','DesignMethod','equiripple','SampleRate',1e3, ...
                 'PassbandFrequency',100,'StopbandFrequency',120, ...
                 'PassbandRipple',0.5,'StopbandAttenuation',60);
iir = designfilt('lowpassiir','DesignMethod','butter','SampleRate',1e3, ...
                 'PassbandFrequency',100,'StopbandFrequency',120, ...
                 'PassbandRipple',0.5,'StopbandAttenuation',60);
FIR = filtord(fir)
IIR = filtord(iir)
```

这是Python filtord.py
```
import numpy as np
from scipy.signal import firwin, iirfilter

def filtord(b, *args):
    if len(args) == 0:
        a = [1]  # Assume FIR for now
    else:
        a = args[0]
    a = np.asarray(a)
    
    # Cast to precision rules
    if np.issubdtype(b.dtype, np.float32) or np.issubdtype(a.dtype, np.float32):
        convClass = np.float32
    else:
        convClass = np.float64
    
    b = b.astype(convClass)
    a = a.astype(convClass)

    # If b is SOS or vector
    if len(args) == 0:
        a1 = a
        if np.ndim(b) == 1:
            # If input is column vector, transpose to obtain row vectors
            if b.ndim == 2 and b.shape[1] == 6:
                # Check if input is a valid SOS matrix
                raise ValueError('Invalid SOS matrix dimensions')
            b1 = b.reshape(1, -1)
        else:
            # If input is a matrix, check if it is a valid SOS matrix
            if b.shape[1] != 6:
                raise ValueError('Invalid SOS matrix dimensions')
            b1, a1 = sos2tf(b)
    else:  # If b and a are vectors
        # If b is not a vector, then only one input is supported
        if b.ndim > 1 and b.shape[0] > 1 and b.shape[1] > 1:
            raise ValueError('Invalid number of inputs')
        
        # If a is not a vector
        if a.ndim > 1 and a.shape[0] > 1 and a.shape[1] > 1:
            raise ValueError('Input not supported')
        
        b1 = b
        a1 = a
        
        
        # If input is column vector, transpose to obtain row vectors
        if b.ndim == 2 and b.shape[1] == 1:
            b1 = b.T
        
        if a.ndim == 2 and a.shape[1] == 1:
            a1 = a.T

    # Normalizing the filter coefficients
    if not np.allclose(b1, 0):
        maxCoefNum = np.max(np.abs(b1))
        b1 /= maxCoefNum
    
    if not np.allclose(a1, 0):
        maxCoefDen = np.max(np.abs(a1))
        a1 /= maxCoefDen
    
    # Returning the index of the last nonzero coefficient
    nZeroLastNum = np.flatnonzero(b1)[-1] if not np.allclose(b1, 0) else 0
    nZeroLastDen = np.flatnonzero(a1)[-1] if not np.allclose(a1, 0) else 0
    
    # Filter order n is maximum of the last nonzero coefficient subtracted by 1
    n = max(nZeroLastNum, nZeroLastDen)
    
    return n


# Placeholder for sos2tf function (not implemented in this example)
def sos2tf(b):
    # Implement SOS to TF conversion if needed
    raise NotImplementedError("sos2tf function is not implemented")
```


### LLM 生成 ： 

下面这是调用程序
```
import numpy as np
from scipy.signal import firwin, iirfilter
from filtord import filtord


# 设计低通 FIR 滤波器（equiripple 方法）
fir_passband_freq = 100  # 通带频率
fir_stopband_freq = 120  # 阻带频率
fir_passband_ripple = 0.5  # 通带波纹
fir_stopband_attenuation = 60  # 阻带衰减

# 计算归一化频率
nyquist_freq = 0.5* 1000  # 采样频率的一半
normalized_passband_freq = fir_passband_freq / nyquist_freq
normalized_stopband_freq = fir_stopband_freq / nyquist_freq

# 计算滤波器系数（FIR）
fir_coeffs = firwin(numtaps=115, cutoff=normalized_passband_freq, \
                         width=None, window='hamming', pass_zero=True, \
                         scale=True, fs=None)
n = filtord(fir_coeffs)
print("Filter order (n):", n)

# 设计低通 IIR 滤波器（Butterworth 方法）
iir_passband_freq = 100  # 通带频率
iir_stopband_freq = 120  # 阻带频率
iir_passband_ripple = 0.5  # 通带波纹
iir_stopband_attenuation = 60  # 阻带衰减

# 计算归一化频率
nyquist_freq = 0.5 * 1000  # 采样频率的一半
normalized_passband_freq = iir_passband_freq / nyquist_freq
normalized_stopband_freq = iir_stopband_freq / nyquist_freq

# 设计滤波器（IIR）
b, a = iirfilter(41, normalized_passband_freq, btype='low', analog=False, ftype='butter', output='ba', fs=1000)
n = filtord(b,a)
print("Filter order (n):", n)
```



