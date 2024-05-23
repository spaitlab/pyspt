# 信号处理仿真与应用 - 数字和模拟滤波器  - 数字滤波器分析

## MATLAB函数描述：stepz 

函数来源：[MATLAB stepz ](https://ww2.mathworks.cn/help/signal/ref/stepz.html)

### 语法

[h,t] = stepz(b,a)
[h,t] = stepz(sos)
[h,t] = stepz(d)
[h,t] = stepz(_,n)
[h,t] = stepz(_,n,fs)
stepz(___)

### 说明

[h，t]=stepz（b，a）返回阶跃响应矢量和数字滤波器的相应采样时间，其中传递函数系数存储在h,t和b,a中
[h，t]=stepz（sos）返回与二阶截面矩阵sos相对应的阶跃响应。                                                                            [h，t]＝stepz（d）返回数字滤波器d的阶跃响应。                                                                                                    [h，t]=stepz（_，n）计算阶跃响应的前n个样本。此语法可以包括来自先前语法的输入参数的任何组合。         [h，t]=stepz（_，n，fs）计算n个样本并产生向量t，使得样本间隔1/fs个单位。                                                                                                          没有输出参数的stepz（___）绘制滤波器的阶跃响应。

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

n — 评估点数
正矢量|正标量                                                                                                                                                                          指定为正整数标量或正整数矢量的求值点数。如果n是正整数标量（t=[0:n-1]'），则函数计算阶跃响应的前n个样本。如果n是整数的矢量，则阶跃响应仅在这些整数值上计算，0表示时间原点。                                                       数据类型:  double

fs — 采样率
正标量                                                                                                                                                                                       采样率，指定为正标量。当时间单位为秒时，fs以赫兹表示。                                                                                                       数据类型:  double

### 输出参量

h — 脉冲响应系数
列向量
阶跃响应，作为列向量返回。如果stepz的输入是单精度的，则函数使用单精度运算来计算阶跃响应。输出h为单精度。

t — 采样时间
列向量
采样时间，以列向量形式返回。



## Python函数描述：stepz

函数来源：自定义

### 滤波器阶跃响应函数定义：

```
import numpy as np
from scipy.signal import lfilter, sosfilt, butter, ellip
from impzlength import impzlength
import matplotlib.pyplot as plt

def stepz(b, *args):
    # Determine if b represents a transfer function or an SOS matrix
    if b.ndim == 1 or b.shape[1] == 1:
        isTF = True
    else:
        isTF = False

    # Process optional arguments
    if len(args) > 0:
        a = args[0]
    else:
        a = np.ones_like(b)

    if len(args) > 1:
        n = int(args[1])
    else:
        n =  None

    if len(args) > 2:
        Fs = float(args[2])
    else:
        Fs = 1.0

    # Compute time vector
    if n is None:
        # Determine the length if not specified
        if isTF:
            N = impzlength(b, a)
        else:
            N = impzlength(b)
        M=0
    elif isinstance(n, (list, np.ndarray)) and len(n) > 1:
        # Vector of indices
        NN = np.round(n).astype(int)
        N = max(NN) + 1
        M = min(NN)
    else:
        # Single value of N
        N = int(n)
        M = 0

    tf = np.arange(M, N) / Fs

    # Form input vector
    x = np.ones_like(tf)

    if isTF:
        sf = lfilter(b, a, x)
    else:
        sf = sosfilt(b, x)

    if isinstance(n, (list, np.ndarray)) and len(n) > 1:
        s = sf[NN - M]
        t = tf[NN - M]
    else:
        s = sf
        t = tf
    # Plotting example using matplotlib
    plt.stem(t, s)
    plt.xlabel('n(Samples)')
    plt.ylabel('Amplitude')
    plt.title('Step Response')
    plt.grid(True)
    plt.show()
    return s, t
```


该函数用于确定数字滤波器的阶跃响应，并可以绘制响应曲线。它利用传递函数系数或SOS矩阵来计算，并且可以处理不同类型的输入参数。

### 参数
- `b`: 滤波器的分子系数，可以是传递函数系数向量或SOS（second-order sections）矩阵。
- *args: 可变参数，用于指定：
  - `a`: 滤波器的分母系数（可选）。
  - `n`: 阶跃响应的点数或点数范围（可选）。
  - `Fs`: 采样频率（可选）。

### 返回值
- `s`: 阶跃响应的幅度数组。
- `t`: 对应的时间点数组。

### 注意事项
- 确保`b`和`a`参数为NumPy数组或可转换为NumPy数组的序列类型。
- 当`b`是SOS矩阵时，确保其形状正确，即每一行有6个元素。
- 如果指定了`n`作为列表或数组，它应该只包含整数。
- `Fs`参数代表采样频率，它决定了时间向量`t`的尺度，应与实际信号的采样频率一致。

### 函数工作原理
1. 函数首先检查`b`是否为SOS矩阵或传递函数系数向量。
2. 根据提供的参数计算阶跃响应长度`N`，如果没有指定，则使用`impzlength`函数来确定。
3. 函数计算阶跃响应，并根据需要绘制阶跃响应的图形。

### 使用场景
- 设计和测试数字滤波器时，用于获取滤波器的阶跃响应，以分析滤波器的性能。
- 在信号处理领域，分析系统对阶跃信号的响应，以了解系统的稳态和瞬态特性。

### 改进建议
- 函数内部的一些辅助函数（如`impzlength`）应该确保被正确导入和可用。
- 可以考虑为`n`和`Fs`参数提供默认值，以简化函数调用。
- 对于复杂的滤波器类型（如SOS），可以增加更多的错误检查，以确保输入数据的正确性。
- 函数可以提供更详细的文档字符串，说明每个参数的作用和预期类型，以及函数的返回值。
- 可以增加测试用例，以验证函数在不同滤波器配置下的正确性和鲁棒性。
- 考虑将绘图部分分离出来，或者提供一个参数来控制是否显示图形，以使函数更加灵活。

## Prompt 1 ： 生成 Python stepz 函数

参考下面MATLAB代码的stepz函数
```
narginchk(1,4)
if ~coder.target('MATLAB')
    if nargout == 0
        % Plotting is not supported for code generation. If this is running in
        % MATLAB, just call MATLAB's stepz, else error.
         coder.internal.assert(coder.target('MEX') || coder.target('Sfun'), ...
            'signal:codegeneration:PlottingNotSupported');
        feval('stepz',b,varargin{:});
        return
    end
end

if coder.internal.isConst(isvector(b)) && isvector(b)
    isTF = true;
    % Checks if B and A are valid numeric data inputs
    validateattributes(b,{'single','double'},{'nonempty'},'stepz','B');
    if nargin > 1
        a = varargin{1};
        coder.internal.assert(isvector(a),'signal:signalanalysisbase:inputnotsupported');
        validateattributes(a,{'single','double'},{'nonempty'},'stepz','A');
    else
        a = ones('like',b);
    end
    eg = zeros(class(a)) + zeros(class(b));
    
    if nargin > 2
        validateattributes(varargin{2},{'numeric'},{'real'},'stepz','N');
        n = double(varargin{2});
    else
        n = [];
    end
    if nargin > 3
        validateattributes(varargin{3},{'numeric'},{'real','scalar'},'stepz','Fs');
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
    validateattributes(b,{'single','double'},{'nonempty','2d'},'stepz','SOS');
    eg = zeros(class(b));
    if nargin > 1
        validateattributes(varargin{1},{'numeric'},{'real'},'stepz','N');
        n = double(varargin{1}); % n can be empty
    else
        n = [];
    end
    if nargin > 2
        validateattributes(varargin{2},{'numeric'},{'real','scalar'},'stepz','Fs');
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
        N = impzlength(b,a);
    else
        N  = impzlength(b);
    end
elseif length(n)>1  % vector of indices
    isVecSpecified = true;
    NN = round(n(:));
    N = max(NN)+1;
    M = min(min(NN),0);
else
    N = n(1);
    isVecSpecified = false;
    NN = 0;
end

tf = cast((M:(N-1))'/Fs,class(eg));

% Form input vector
x = ones(size(tf),class(eg));

if isTF
    sf = filter(b,a,x);
else
    sf = sosfilt(b,x);
end

if isVecSpecified
    s = sf(NN-M+1);
    t = tf(NN-M+1);
else
    s = sf;
    t = tf;
end

if nargout
    varargout{1} = s;
    varargout{2} = t;
else
    timezplot(t,s,Fs,getString(message('signal:stepz:Step')));
end
```

这个函数的功能是生成滤波器的阶跃响应，输入参数为滤波器的b，a值。请根据这个函数完成Python语言stepz函数的编写，便于类似应用调用。

### LLM 生成 ： Python stepz 函数

这是该函数的定义：

    import numpy as np
    from scipy.signal import lfilter, sosfilt, butter, ellip
    from impzlength import impzlength
    import matplotlib.pyplot as plt
    
    def stepz(b, *args):
        # Determine if b represents a transfer function or an SOS matrix
        if b.ndim == 1 or b.shape[1] == 1:
            isTF = True
        else:
            isTF = False
    
        # Process optional arguments
        if len(args) > 0:
            a = args[0]
        else:
            a = np.ones_like(b)
    
        if len(args) > 1:
            n = int(args[1])
        else:
            n =  None
    
        if len(args) > 2:
            Fs = float(args[2])
        else:
            Fs = 1.0
    
        # Compute time vector
        if n is None:
            # Determine the length if not specified
            if isTF:
                N = impzlength(b, a)
            else:
                N = impzlength(b)
            M=0
        elif isinstance(n, (list, np.ndarray)) and len(n) > 1:
            # Vector of indices
            NN = np.round(n).astype(int)
            N = max(NN) + 1
            M = min(NN)
        else:
            # Single value of N
            N = int(n)
            M = 0
    
        tf = np.arange(M, N) / Fs
    
        # Form input vector
        x = np.ones_like(tf)
    
        if isTF:
            sf = lfilter(b, a, x)
        else:
            sf = sosfilt(b, x)
    
        if isinstance(n, (list, np.ndarray)) and len(n) > 1:
            s = sf[NN - M]
            t = tf[NN - M]
        else:
            s = sf
            t = tf
        # Plotting example using matplotlib
        plt.stem(t, s)
        plt.xlabel('n(Samples)')
        plt.ylabel('Amplitude')
        plt.title('Step Response')
        plt.grid(True)
        plt.show()
        return s, t


## Prompt 2 ： 应用生成的 Python stepz.py函数做MATLAB示例对应

这是MATLAB程序1，要求创建一个标准化半功率频率为0.4πrad/sample的三阶巴特沃斯滤波器。显示其阶跃响应。

```
[b,a] = butter(3,0.4);
stepz(b,a)
```

这是stepz函数定义

    import numpy as np
    from scipy.signal import lfilter, sosfilt, butter, ellip
    from impzlength import impzlength
    import matplotlib.pyplot as plt
    
    def stepz(b, *args):
        # Determine if b represents a transfer function or an SOS matrix
        if b.ndim == 1 or b.shape[1] == 1:
            isTF = True
        else:
            isTF = False
    
        # Process optional arguments
        if len(args) > 0:
            a = args[0]
        else:
            a = np.ones_like(b)
    
        if len(args) > 1:
            n = int(args[1])
        else:
            n =  None
    
        if len(args) > 2:
            Fs = float(args[2])
        else:
            Fs = 1.0
    
        # Compute time vector
        if n is None:
            # Determine the length if not specified
            if isTF:
                N = impzlength(b, a)
            else:
                N = impzlength(b)
            M=0
        elif isinstance(n, (list, np.ndarray)) and len(n) > 1:
            # Vector of indices
            NN = np.round(n).astype(int)
            N = max(NN) + 1
            M = min(NN)
        else:
            # Single value of N
            N = int(n)
            M = 0
    
        tf = np.arange(M, N) / Fs
    
        # Form input vector
        x = np.ones_like(tf)
    
        if isTF:
            sf = lfilter(b, a, x)
        else:
            sf = sosfilt(b, x)
    
        if isinstance(n, (list, np.ndarray)) and len(n) > 1:
            s = sf[NN - M]
            t = tf[NN - M]
        else:
            s = sf
            t = tf
        # Plotting example using matplotlib
        plt.stem(t, s)
        plt.xlabel('n(Samples)')
        plt.ylabel('Amplitude')
        plt.title('Step Response')
        plt.grid(True)
        plt.show()
        return s, t

请应用自定义stepz函数，实现上述MATLAB程序的功能，并给出Python程序。

### LLM 生成 ： 
```
import numpy as np
from scipy.signal import lfilter, sosfilt, butter, ellip
from impzlength import impzlength
import matplotlib.pyplot as plt
from stepz import stepz

# 示例1
# 设计三阶巴特沃斯低通滤波器
order = 3  # 滤波器阶数
cutoff_freq = 0.4  # 截止频率
b, a = butter(order, cutoff_freq, btype='low')  # 设计低通滤波器
s,t = stepz(b, a)
```

## Prompt 3 ： 应用生成的 Python stepz.py 函数做MATLAB示例对应

这是MATLAB程序2，要求设计一个归一化通带频率为0.4πrad/sample的四阶低通椭圆滤波器。指定0.5 dB的通带纹波和20 dB的阻带衰减。绘制滤波器阶跃响应的前50个样本。参考下面Matlab代码，给出Python实现代码：

**Matlab代码**

```
[b,a] = ellip(4,0.5,20,0.4);
stepz(b,a,50)
```

这是Python stepz.py
```
import numpy as np
from scipy.signal import lfilter, sosfilt, butter, ellip
from impzlength import impzlength
import matplotlib.pyplot as plt

def stepz(b, *args):
    # Determine if b represents a transfer function or an SOS matrix
    if b.ndim == 1 or b.shape[1] == 1:
        isTF = True
    else:
        isTF = False

    # Process optional arguments
    if len(args) > 0:
        a = args[0]
    else:
        a = np.ones_like(b)

    if len(args) > 1:
        n = int(args[1])
    else:
        n =  None

    if len(args) > 2:
        Fs = float(args[2])
    else:
        Fs = 1.0

    # Compute time vector
    if n is None:
        # Determine the length if not specified
        if isTF:
            N = impzlength(b, a)
        else:
            N = impzlength(b)
        M=0
    elif isinstance(n, (list, np.ndarray)) and len(n) > 1:
        # Vector of indices
        NN = np.round(n).astype(int)
        N = max(NN) + 1
        M = min(NN)
    else:
        # Single value of N
        N = int(n)
        M = 0

    tf = np.arange(M, N) / Fs

    # Form input vector
    x = np.ones_like(tf)

    if isTF:
        sf = lfilter(b, a, x)
    else:
        sf = sosfilt(b, x)

    if isinstance(n, (list, np.ndarray)) and len(n) > 1:
        s = sf[NN - M]
        t = tf[NN - M]
    else:
        s = sf
        t = tf
    # Plotting example using matplotlib
    plt.stem(t, s)
    plt.xlabel('n(Samples)')
    plt.ylabel('Amplitude')
    plt.title('Step Response')
    plt.grid(True)
    plt.show()
    return s, t
```


### LLM 生成 ： 

下面这是调用程序
```
import numpy as np
from scipy.signal import lfilter, sosfilt, butter, ellip
from impzlength import impzlength
import matplotlib.pyplot as plt
from stepz import stepz

# 示例2
# 设计椭圆低通滤波器
N = 4  # 滤波器阶数
rp = 0.5  # 通带最大允许波纹（dB）
rs = 20  # 阻带最小衰减（dB）
cutoff = 0.4  # 截止频率（归一化频率，范围为0到1）
# 设计椭圆滤波器
b, a = ellip(N, rp, rs, cutoff, output='ba')
N=50
s,t = stepz(b, a, 50)
```



