# 信号处理仿真与应用 - 数字和模拟滤波器  - 数字滤波器分析

## MATLAB函数描述：impz 

函数来源：[MATLAB impz](https://ww2.mathworks.cn/help/signal/ref/impz.html)

### 语法

[h,t] = impz(b,a)
[h,t] = impz(sos)
[h,t] = impz(d)
[h,t] = impz(,n)
[h,t] = impz(,n,fs)
impz(___)

### 说明

[h,t] = impz（b,a） 返回数字滤波器的脉冲响应，分子系数和分母系数。函数选择样本数并返回 in 中的响应系数和 in 中的采样时间。
[h,t] = impz（sos） 返回由二阶截面矩阵指定的滤波器的脉冲响应。
[h,t] = impz（d） 返回数字滤波器的脉冲响应。使用 designfilt 可以根据频率响应规范生成。
[h,t] = impz(,n)指定要计算的脉冲响应样本。可以使用任何前面的语法。
[h,t] = impz(,n,fs)返回一个连续样本间隔为 1/fs 单位的向量。
impz(___)在没有输出参数的情况下绘制脉冲滤波器的响应。

### 输入参数

b,a  — 传递函数系数
向量 
传递函数系数，指定为向量。将传递函数以a,b的形式表达出来，传递函数形式如下：
$$
H(z)=\frac{B(Z)}{A(Z)}=\frac{b_{1}+b_{2}z^{-1}...+b_{n+1}z^{-n}}{a_{1}+a_{2}z^{-1}...+a_{m+1}z^{-m}} 
$$
示例：指定归一化 3 dB 频率为 0.5π rad/sample的三阶巴特沃斯滤波器 。`b = [1 3 3 1]/6``a = [3 0 1 0]/3`
复数支持：是`double``single`

sos — 二阶截面系数
矩阵
二阶截面系数，指定为矩阵。 是一个 *K*×6 矩阵，其中截面数 *K* 必须大于或等于2。如果节数小 2，该函数将输入视为分子向量。每行对应于二阶的系数滤波器。第 *i*行对应于 `sos``[bi(1) bi(2) bi(3) ai(1) ai(2) ai(3)]`
示例：指定三阶巴特沃斯 滤波器，归一化 3 dB 频率 0.5π rad/样本。`s = [2 4 2 6 0 2;3 3 0 6 0 0]`	  复数支持：是`double``single`

d — 数字滤波器
digitalFilter对象数字筛选器，指定为 [`digitalFilter`](https://ww2.mathworks.cn/help/signal/ref/digitalfilter.html) 对象。使用 [`designfilt`](https://ww2.mathworks.cn/help/signal/ref/designfilt.html) 生成数字滤波器 基于频率响应规范。                                                                                                                                                                                     示例：指定归一化 3 dB 频率为 0.5π 的三阶巴特沃斯滤波器 rad/sample。`d = designfilt('lowpassiir','FilterOrder',3,'HalfPowerFrequency',0.5)`

n — 样本数
正整数|非负整数向量|[ ]                                                                                                                                                   样本数，指定为正整数，非负整数的向量， 或空向量。

- 如果为正整数`n`，则计算脉冲响应的第一个样本，并返回 [`t`](https://ww2.mathworks.cn/help/signal/ref/impz.html#mw_948c6ded-2a5c-4858-850e-bf1031a9dd29) 作为 `(0:n-1)'。
- 如果是非负整数的向量，则计算以下位置的脉冲响应在向量中指定。
- 如果为空向量，则自动计算样本数。有关详细信息，请参阅[算法](https://ww2.mathworks.cn/help/signal/ref/impz.html#mw_4a2ac3d4-87ad-4f9a-ba46-a94c500c0d15)信息。

示例：计算前五个巴特沃斯滤波器的脉冲响应样本。`impz([2 4 2 6 0 2;3 3 0 6 0 0],5)`                             示例：计算巴特沃斯滤波器脉冲响应的前六个样本。`impz([2 4 2 6 0 2;3 3 0 6 0 0],[0 3 2 1 4 5])`     示例：计算巴特沃斯滤波器的脉冲响应，设计用于滤波 5 处采样的信号。`impz([2 4 2 6 0 2;3 3 0 6 0 0],[],5e3)`

fs — 采样率                                                                                                                                                                         采样率，指定为正标量。当时间单位为秒时，以赫兹表示。                                                                                         数据类型：`double`

### 输出参量

h — 脉冲响应系数
列向量
脉冲响应系数，以列向量形式返回。

t — 采样时间
列向量
采样时间，以列向量形式返回。



## Python函数描述：impz

函数来源：自定义

### 滤波器脉冲响应函数定义：

```
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def impz(b, a=None, *args):
    if a is None:
        # b is SOS matrix or single vector (transfer function)
        if np.ndim(b) == 2 and np.size(b, 1) == 6:
            isTF = False  # SOS matrix
            sos = b
        else:
            isTF = True  # single transfer function vector
            sos = None  # not used in this case
        
        if len(args) > 0:
            n = int(args[0])
        else:
            n = None
        
        if len(args) > 1:
            Fs = float(args[1])
        else:
            Fs = 1.0
    
    else:
        # b and a are coefficient vectors (transfer function)
        isTF = True
        sos = None
        
        n = args[0] if len(args) > 0 else None
        Fs = args[1] if len(args) > 1 else 1.0
    
    # Compute impulse response
    if n is None:
        # Determine length of impulse response automatically
        if isTF:
            impulse_response = signal.impulse((b, a), N=1000)
        else:
            impulse_response = signal.sosfilt(sos, np.array([1] + [0] * 999))
    else:
        # User specified length of impulse response
        impulse_response = np.zeros(n + 1)
        impulse_response[0] = 1
        if isTF:
            impulse_response = signal.lfilter(b, a, impulse_response)
        else:
            impulse_response = signal.sosfilt(sos, impulse_response)
    
    # Compute time vector
    t = np.arange(len(impulse_response)) / Fs
    
    return impulse_response, t
```


这段代码定义了一个名为 `impz` 的函数，其目的是计算数字滤波器的脉冲响应。脉冲响应是分析数字滤波器性能的重要工具，它描述了系统对单位脉冲信号的响应。该函数接受滤波器的分子系数 `b` 和分母系数 `a` 作为输入，还可以接收额外的参数来指定脉冲响应的点数 `n` 和采样频率 `Fs`。函数内部使用 `scipy.signal` 模块中的函数来计算脉冲响应，并根据需要绘制脉冲响应的图形。

### 参数
- `b`: 滤波器的分子系数，当`a`参数未指定时，`b`可以是一个传递函数向量或者一个SOS（second-order sections）矩阵。
- `a`: 滤波器的分母系数，可选参数，当指定时，`b`和`a`共同构成滤波器的传递函数系数向量。
- `*args`: 可变参数，用于指定脉冲响应的点数`n`和采样频率`Fs`。

### 返回值
- `impulse_response`: 脉冲响应的数组。
- `t`: 时间向量，表示脉冲响应对应的时间点。

### 注意事项
- 函数没有内置的错误处理机制，因此在传递不合适的参数时可能会导致运行时错误。建议在实际应用中添加适当的错误检查和异常处理。

### 函数工作原理
1. 函数首先检查是否提供了`a`参数。如果没有提供`a`，则根据`b`的形状判断它是SOS矩阵还是传递函数向量。
2. 根据提供的参数，函数计算脉冲响应。如果用户没有指定脉冲响应的长度，则函数会自动确定一个合适的长度（1000点）。
3. 使用`scipy.signal.impulse`、`signal.lfilter`或`signal.sosfilt`计算脉冲响应，这取决于输入参数的类型。
4. 最后，函数根据采样频率计算时间向量，并返回脉冲响应及其对应的时间点。

### 使用场景
- 设计和测试数字滤波器时，用于获取滤波器的脉冲响应，以分析滤波器的性能。
- 在信号处理领域，分析系统对单位脉冲信号的响应，以了解系统的时域特性。

### 改进建议
- 函数可以提供更详细的文档字符串（docstring），说明每个参数的作用和预期类型，以及函数的返回值。
- 对于可选参数`n`和`Fs`，可以提供默认值，以简化函数调用。
- 可以增加参数检查，确保提供的参数是有效的，比如`b`和`a`是正确的长度和类型。
- 考虑增加对其他类型的滤波器表示（如ZPK格式）的支持。

## Prompt 1 ： 生成 Python impz 函数

参考下面MATLAB代码的impz函数
```
narginchk(1,4)

if ~coder.target('MATLAB')
    if nargout == 0
        % Plotting is not supported for code generation. If this is running in
        % MATLAB, just call MATLAB's impz, else error.
         coder.internal.assert(coder.target('MEX') || coder.target('Sfun'), ...
            'signal:codegeneration:PlottingNotSupported');
        feval('impz',b,varargin{:});
        return
    end
end

if coder.internal.isConst(isvector(b)) && isvector(b)
    isTF = true; % transfer function
    % Checks if B and A are valid numeric data inputs
    validateattributes(b,{'single','double'},{},'impz','B');
    if nargin > 1
        a = varargin{1};
        coder.internal.assert(isvector(a),'signal:signalanalysisbase:inputnotsupported');
        validateattributes(a,{'single','double'},{},'impz','A');
    else
        a = ones('like',b);
    end
    eg  = zeros(class(b)) + zeros(class(a));
    if nargin > 2
        validateattributes(varargin{2},{'numeric'},{'real'},'impz','N');
        n = double(varargin{2});
    else
        n = [];
    end
    if nargin > 3
        validateattributes(varargin{3},{'numeric'},{'real','scalar'},'impz','Fs');
        Fs = double(varargin{3}(1));
    else
        Fs = 1;
    end
else
    % error out if the variable-sized matrix input b becomes a vector at runtime
    if isvector(b)
        coder.internal.error('signal:signalanalysisbase:varSizeMatrixCannotBecomeVector')
    end
    % Input is a matrix, check if it is a valid SOS matrix
    coder.internal.assert(size(b,2) == 6,...
        'signal:signalanalysisbase:invalidinputsosmatrix');
    isTF = false; % SOS instead of transfer function
    % Checks if SOS is a valid numeric data input
    validateattributes(b,{'single','double'},{'nonempty','2d'},'impz','SOS');
    eg  = zeros(class(b)) ;
    if nargin > 1
        validateattributes(varargin{1},{'numeric'},{'real'},'impz','N');
        n = double(varargin{1}); % n can be empty
    else
        n = [];
    end
    if nargin > 2
        validateattributes(varargin{2},{'numeric'},{'real','scalar'},'impz','Fs');
        Fs = double(varargin{2}(1));
    else
        Fs = 1;
    end
end

% Compute time vector
M = 0;
if isempty(n)
    % if not specified, determine the length
    NN = 0;
    isVecSpecified = false;
    if isTF
        N = impzlength(b,a,.00005);
    else
        N  = impzlength(b,.00005);
    end
elseif length(n)>1  % vector of indices
    isVecSpecified = true;
    NN = round(n(:));
    N = max(NN)+1;
    M = min(min(NN),0);
else % impulse reponse length is specified
    N = n(1);
    isVecSpecified = false;
    NN = 0;
end

tf = cast((M:(N-1))',class(eg));
% Filter an impulse with input coefficients
x  = cast(tf == 0,class(eg));

if isTF
    hf = filter(b,a,x);
else
    hf = sosfilt(b,x);
end

if isVecSpecified
    h = hf(NN-M+1);
    t = tf(NN-M+1);
else
    h = hf;
    t = tf;
end

t = t/Fs;

if nargout
    varargout{1} = h;
    varargout{2} = t;
else
    timezplot(t,h,Fs,'Impulse');
end
```

这个函数的功能是生成滤波器的冲激响应，输入参数为滤波器的b，a值，以及采样点数。请根据这个函数完成Python语言impz函数的编写，便于类似应用调用。

### LLM 生成 ： Python impz 函数

这是该函数的定义：

    import numpy as np
    import scipy.signal as signal
    import matplotlib.pyplot as plt
    
    def impz(b, a=None, *args):
        if a is None:
            # b is SOS matrix or single vector (transfer function)
            if np.ndim(b) == 2 and np.size(b, 1) == 6:
                isTF = False  # SOS matrix
                sos = b
            else:
                isTF = True  # single transfer function vector
                sos = None  # not used in this case
            
            if len(args) > 0:
                n = int(args[0])
            else:
                n = None
            
            if len(args) > 1:
                Fs = float(args[1])
            else:
                Fs = 1.0
        
        else:
            # b and a are coefficient vectors (transfer function)
            isTF = True
            sos = None
            
            n = args[0] if len(args) > 0 else None
            Fs = args[1] if len(args) > 1 else 1.0
        
        # Compute impulse response
        if n is None:
            # Determine length of impulse response automatically
            if isTF:
                impulse_response = signal.impulse((b, a), N=1000)
            else:
                impulse_response = signal.sosfilt(sos, np.array([1] + [0] * 999))
        else:
            # User specified length of impulse response
            impulse_response = np.zeros(n + 1)
            impulse_response[0] = 1
            if isTF:
                impulse_response = signal.lfilter(b, a, impulse_response)
            else:
                impulse_response = signal.sosfilt(sos, impulse_response)
        
        # Compute time vector
        t = np.arange(len(impulse_response)) / Fs
        
        return impulse_response, t


## Prompt 2 ： 应用生成的 Python impz.py函数做MATLAB示例对应

这是MATLAB程序1，要求为设计一个归一化通带频率为0.4 rad/样本的四阶低通椭圆滤波器。指定 0.5 dB 的通带纹波和 20 dB 的阻带衰减。绘制脉冲响应的前 50 个样本。

```
[b,a] = ellip(4,0.5,20,0.4);
impz(b,a,50)
```

这是impz函数定义

    import numpy as np
    import scipy.signal as signal
    import matplotlib.pyplot as plt
    
    def impz(b, a=None, *args):
        if a is None:
            # b is SOS matrix or single vector (transfer function)
            if np.ndim(b) == 2 and np.size(b, 1) == 6:
                isTF = False  # SOS matrix
                sos = b
            else:
                isTF = True  # single transfer function vector
                sos = None  # not used in this case
            
            if len(args) > 0:
                n = int(args[0])
            else:
                n = None
            
            if len(args) > 1:
                Fs = float(args[1])
            else:
                Fs = 1.0
        
        else:
            # b and a are coefficient vectors (transfer function)
            isTF = True
            sos = None
            
            n = args[0] if len(args) > 0 else None
            Fs = args[1] if len(args) > 1 else 1.0
        
        # Compute impulse response
        if n is None:
            # Determine length of impulse response automatically
            if isTF:
                impulse_response = signal.impulse((b, a), N=1000)
            else:
                impulse_response = signal.sosfilt(sos, np.array([1] + [0] * 999))
        else:
            # User specified length of impulse response
            impulse_response = np.zeros(n + 1)
            impulse_response[0] = 1
            if isTF:
                impulse_response = signal.lfilter(b, a, impulse_response)
            else:
                impulse_response = signal.sosfilt(sos, impulse_response)
        
        # Compute time vector
        t = np.arange(len(impulse_response)) / Fs
        
        return impulse_response, t

请应用自定义impz函数，实现上述MATLAB程序的功能，并给出Python程序。

### LLM 生成 ： 
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin, ellip
from impz import impz

# 使用椭圆低通滤波器的示例
N = 4  # 滤波器阶数
rp = 0.5  # 通带纹波(dB)
rs = 20  # 阻带衰减(dB)
fc = 0.4  # 归一化通带频率

# 设计椭圆低通滤波器，返回分子系数 b 和分母系数 a
b, a = ellip(N, rp, rs, fc, 'low', analog=False, output='ba')
num_samples = 50

h, t = impz(b, a, num_samples, 1.0)

# 绘制脉冲响应图
plt.stem(t, h)
plt.title('Impulse Response (Elliptic Lowpass Filter)')
plt.xlabel('n (samples)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```

## Prompt 3 ： 应用生成的 Python impz.py 函数做MATLAB示例对应

这是MATLAB程序2，要求使用 Kaiser 窗口设计 18 阶的 FIR 高通滤波器，其中*β*=4，指定 100 Hz 的采样率和 30 Hz 的截止频率，显示滤波器的脉冲响应。参考下面Matlab代码，给出Python实现代码：

**Matlab代码**

```
b = fir1(18,30/(100/2),'high',kaiser(19,4));
impz(b,1,[],100)
```

这是Python impz.py
```
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def impz(b, a=None, *args):
    if a is None:
        # b is SOS matrix or single vector (transfer function)
        if np.ndim(b) == 2 and np.size(b, 1) == 6:
            isTF = False  # SOS matrix
            sos = b
        else:
            isTF = True  # single transfer function vector
            sos = None  # not used in this case
        
        if len(args) > 0:
            n = int(args[0])
        else:
            n = None
        
        if len(args) > 1:
            Fs = float(args[1])
        else:
            Fs = 1.0
    
    else:
        # b and a are coefficient vectors (transfer function)
        isTF = True
        sos = None
        
        n = args[0] if len(args) > 0 else None
        Fs = args[1] if len(args) > 1 else 1.0
    
    # Compute impulse response
    if n is None:
        # Determine length of impulse response automatically
        if isTF:
            impulse_response = signal.impulse((b, a), N=1000)
        else:
            impulse_response = signal.sosfilt(sos, np.array([1] + [0] * 999))
    else:
        # User specified length of impulse response
        impulse_response = np.zeros(n + 1)
        impulse_response[0] = 1
        if isTF:
            impulse_response = signal.lfilter(b, a, impulse_response)
        else:
            impulse_response = signal.sosfilt(sos, impulse_response)
    
    # Compute time vector
    t = np.arange(len(impulse_response)) / Fs
    
    return impulse_response, t
```


### LLM 生成 ： 

下面这是调用程序
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin, ellip
from impz import impz

# 使用 FIR 高通滤波器的示例
order = 18  # 滤波器阶数
fs = 100  # 采样率 (Hz)
cutoff_freq = 30  # 截止频率 (Hz)
beta = 4  # Kaiser 窗口的β值

# 使用 firwin 函数设计 FIR 高通滤波器
b = firwin(order + 1, cutoff_freq / (fs / 2), pass_zero=False, window=('kaiser', beta))
num_samples = 100

# 使用 calculate_impulse_response 函数计算滤波器的脉冲响应和时间序列
h, t = impz(b, 1, num_samples, 1.0)

# 绘制脉冲响应图
plt.stem(t * 10, h)
plt.title('Impulse Response (FIR Highpass Filter)')
plt.xlim(0, 180)
plt.xlabel('nT (ms)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```



