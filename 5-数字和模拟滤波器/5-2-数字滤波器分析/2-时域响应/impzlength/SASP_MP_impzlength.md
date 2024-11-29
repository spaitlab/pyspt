# 信号处理仿真与应用 - 数字和模拟滤波器  - 数字滤波器分析

## MATLAB函数描述：impzlength 

函数来源：[MATLAB impzlength](https://ww2.mathworks.cn/help/signal/ref/impzlength.html)

### 语法

len = impzlength(b,a)
len = impzlength(sos)
len = impzlength(d)
len = impzlength(___,tol)

### 说明

len=impzlength（b，a）返回因果离散时间滤波器的脉冲响应长度，该滤波器具有由分子和分母多项式指定的有理系统函数，单位为baz<sup>*-1*</sup>。对于稳定的IIR滤波器，是有效脉冲响应序列长度。IIR滤波器的脉冲响应中第len项之后的项基本上为零。
len=impzlength（sos）返回由二阶分段矩阵sos指定的IIR滤波器的有效脉冲响应长度。sos是一个K-by-6矩阵，其中节数K必须大于或等于2。如果节的数量小于2，impzlength将输入视为分子向量b。每行sos对应于二阶（双四阶）滤波器的系数。sos矩阵的第i行对应于[bi（1）bi（2）bi（3）ai（1）ai（2）ai（3）]。
len=impzlength（d）返回数字滤波器的脉冲响应长度。使用designfilt根据频率响应规范生成d。
len=impzlength（，tol）指定用于估计IIR滤波器脉冲响应的有效长度的容差。默认情况下，tol为5e-5。增加tol的值估计IIR滤波器的脉冲响应的有效长度更短。降低tol的值会使IIR滤波器的脉冲响应产生更长的有效长度。

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

tol — IIR滤波器有效脉冲响应长度公差
5e-5（默认）|正标量                                                                                                                                                        IIR滤波器有效脉冲响应长度的容差，指定为正数。公差决定了绝对可和序列中的项，在该序列之后，后续项被认为是0。默认公差为5e-5。增加容差将返回更短的有效脉冲响应序列长度。减小公差将返回更长的有效脉冲响应序列长度。

### 输出参量

len — 脉冲响应的长度
正整数
脉冲响应的长度，指定为正整数。对于具有绝对可和脉冲响应的稳定IIR滤波器，impzlength返回脉冲响应的有效长度，超过该长度，系数基本为零。您可以通过指定可选的tol输入参数来控制这个截止点。



## Python函数描述：impzlength

函数来源：自定义

### 滤波器脉冲响应函数定义：

```
import numpy as np
from scipy import signal
from scipy.signal import ellip, zpk2sos, firwin, butter

def impzlength(b, *args):
    if isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[1] == 6:
        # SOS 矩阵情况
       # if not np.isclose(np.linalg.norm(b), 0.0):
       #     raise ValueError("参数 'b' 必须是非空的 SOS 矩阵.")

        if len(args) > 0:
            tol = args[0]
        else:
            tol = 0.00005
        return lclsosimpzlength(b, tol)
    else:
        # 传递函数情况
        isTF = True
        if len(args) == 0:
            a = np.array([1.0])  # 传递函数的默认 'a'
        else:
            a = args[0]

        if len(args) < 2:
            tol = 0.00005
        else:
            tol = float(args[1])

        return calculate_impz_length(b, a, tol, isTF)

def calculate_impz_length(b, a, tol, isTF):
    if isTF:
        # 判断滤波器是否为 FIR（所有 'a' 的元素接近于零）
        if np.allclose(a, 0.0, atol=1e-15):
            return len(b)  # FIR 滤波器的长度
        else:
            # 找到向量 b 中第一个非零元素的索引，作为延迟 delay
            delay = find_first_nonzero_index(b)

            # 根据稳定性和延迟计算长度
            p = np.roots(a)
            if np.any(np.abs(p) > 1.0001):
                N = unstable_length(p)
            else:
                N = stableNmarginal_length(p, tol, delay)

            return int(np.ceil(N))-1  # 确保 N 是整数并向上取整
    else:
        return lclsosimpzlength(b, tol)

def find_first_nonzero_index(arr):
    # 找到数组 arr 中第一个非零元素的索引
    indices = np.nonzero(arr)[0]  # 找到所有非零元素的索引
    if len(indices) == 0:
        return 0  # 如果数组中没有非零元素，返回索引 0（默认延迟为 0）
    else:
        return indices[0]  # 返回第一个非零元素的索引

def unstable_length(p):
    ind = np.abs(p) > 1
    return int(6 / np.log10(np.max(np.abs(p[ind]))))

def stableNmarginal_length(p, tol, delay):
    n = len(p)
    nOscillation = 0
    indOscillation = []
    nDamped = 0
    indDamped = []

    # 遍历极点 p
    for i in range(n):
        # 如果极点接近于1，则取其相反数
        if np.abs(p[i] - 1) < 1e-5:
            p[i] = -p[i]
        
        # 判断极点是振荡还是非振荡
        if np.abs(np.abs(p[i]) - 1) < 1e-5:
            nOscillation += 1
            indOscillation.append(i)
        else:
            nDamped += 1
            indDamped.append(i)

    if nOscillation == n:  # 全部为振荡极点
        periods = 5 * np.max(2 * np.pi / np.abs(np.angle(p)))
        N = periods
    elif nOscillation == 0:  # 全部为非振荡极点
        maxp = np.max(np.abs(p))
        maxind = np.argmax(np.abs(p))
        multiplicity = mltplcty(p, maxind, tol)
        N = multiplicity * np.log10(tol) / np.log10(maxp) + delay
    else:  # 部分振荡部分非振荡
        pOscillation = p[indOscillation]
        pDamped = p[indDamped]
        periods = 5 * np.max(2 * np.pi / np.abs(np.angle(pOscillation)))
        maxp = np.max(np.abs(pDamped))
        maxind = np.argmax(np.abs(pDamped))
        multiplicity = mltplcty(pDamped, maxind, tol)
        N = max(periods, multiplicity * np.log10(tol) / np.log10(maxp)) + delay

    return N

def mltplcty(p, ind, tol=0.001):
    if np.any(p == 0):
        thresh = np.float(tol)
    else:
        thresh = tol * np.abs(p[ind])
    
    m = 0
    for i in range(len(p)):
        if np.abs(p[i] - p[ind]) < thresh:
            m += 1
    
    return m

def lclsosimpzlength(sos, tol):
    firlen = 1
    iirlen = 1
    num_sections = sos.shape[0]
    for k in range(num_sections):
        b = sos[k, 0:3]
        a = sos[k, 3:6]
        if np.allclose(a, 0.0, atol=1e-15):
            return len(b)  # FIR 滤波器的长度
        else:
            iirlen = max(iirlen, impzlength(b, a, tol))
    return max(firlen, iirlen)
```


该函数用于确定数字滤波器的脉冲响应所需的样本点数，这取决于滤波器的稳定性和延迟。

### 参数
- `b`: 滤波器的分子系数，可以是传递函数系数向量或SOS（second-order sections）矩阵。
- `*args`: : 可变参数，用于指定：
  - 当`b`是SOS矩阵时，`args[0]`为公差`tol`，用于判断极点稳定性。
  - 当`b`是传递函数系数向量时，`args[0]`为分母系数`a`，`args[1]`为公差`tol`。

### 返回值
- 脉冲响应的长度`N`，作为整数返回。

### 注意事项
- 确保`b`和`a`参数为NumPy数组或可转换为NumPy数组的序列类型。
- 当`b`是SOS矩阵时，确保其形状正确，即每一行有6个元素。
- 当指定`a`时，`b`和`a`的长度应该匹配。
- `tol`参数用于判断极点的稳定性，其值应适当选择以确保准确性。

### 函数工作原理
1. 函数首先检查`b`是否为SOS矩阵。如果是，使用特定的方法计算长度。
2. 如果`b`是传递函数系数向量，函数会检查`a`是否几乎为零（即FIR滤波器），然后根据滤波器的稳定性和延迟计算脉冲响应长度。
3. 对于IIR滤波器，函数会分析极点的位置和类型（振荡或非振荡）来确定长度。

### 使用场景
- 设计和测试数字滤波器时，用于估计计算脉冲响应所需的样本点数。
- 在信号处理领域，分析系统对单位脉冲信号的响应时长。

### 改进建议
- 函数内部的一些辅助函数（如`find_first_nonzero_index`、`unstable_length`等）可以进一步文档化，以说明它们的作用和预期行为。
- 可以考虑为`tol`参数提供一个合理的默认值，以简化函数调用。
- 对于复杂的滤波器类型（如SOS），可以增加更多的错误检查，以确保输入数据的正确性。
- 函数可以提供更详细的文档字符串，说明每个参数的作用和预期类型，以及函数的返回值。
- 可以增加测试用例，以验证函数在不同滤波器配置下的正确性和鲁棒性。

## Prompt 1 ： 生成 Python impzlength 函数

参考下面MATLAB代码的impzlength函数
```
narginchk(1,3)

% transfer function
if coder.internal.isConst(isvector(b)) && isvector(b)
    isTF = true;
    if nargin == 1
        a = 1;
    else
        a = varargin{1};
    end
    % Checks if B and A are valid numeric data inputs
    validateattributes(a,{'double','single'},{'nonempty'},'impzlength','A');
    validateattributes(b,{'double','single'},{'nonempty'},'impzlength','B');
    if nargin < 3
        tol = .00005;
    else
        validateattributes(varargin{2},{'numeric'},{'scalar','real','positive'},'impzlength','TOL');
        % Cast to enforce precision rules
        tol = double(varargin{2}(1));
    end    
else
    % Checks if SOS is a valid numeric data input
    validateattributes(b,{'single','double'},{'nonempty'},'impzlength','SOS');  
    
    % error out if a variable-sized matrix becomes a vector at runtime
    if isvector(b)
        coder.internal.error('signal:signalanalysisbase:varSizeMatrixCannotBecomeVector')
    end    
    % Input is a matrix, check if it is a valid SOS matrix
    coder.internal.assert(size(b,2) == 6,...
                        'signal:signalanalysisbase:invalidinputsosmatrix');  
    isTF = false; % SOS instead of transfer function

    if nargin > 1
        validateattributes(varargin{1},{'numeric'},{'scalar','real','positive'},'impzlength','TOL');
        % Cast to enforce precision rules
        tol = double(varargin{1}(1));
    else
        tol = .00005;
    end
end

if isTF    
    % Determine if filter is FIR
    if signal.internal.isfir(b,a)
        N = length(b);
    else
        indx = find(b, 1);
        if isempty(indx)
            delay = 0;
        else
            delay=indx(1)-1;
        end
        p = roots(a(:));
        if any(abs(p) > 1.0001)
            N = unstable_length(p);
        else
            N = stableNmarginal_length(p,tol,delay);
        end
        % Cast to enforce precision rules
        N = double(N);
        N = max(length(a)+length(b)-1,N);
        
        % Always return an integer length
        N = floor(N);
    end
else
    N = lclsosimpzlength(b,tol);
    % Cast to enforce precision rules
    N = double(N);
end
%-------------------------------------------------------------------------
function N = unstable_length(p)
% Determine the length for an unstable filter
ind = abs(p)>1;
N = 6/log10(max(abs(p(ind))));% 1000000 times original amplitude


%-------------------------------------------------------------------------
function N = stableNmarginal_length(p,tol,delay)
% Determine the length for an unstable filter
n = length(p);
nOscillation = 0;
indOscillation = zeros(n,1);
nDamped = 0;
indDamped = zeros(n,1);
for i = 1:n
    %minimum height is .00005 original amplitude:
    if abs(p(i)-1) < 1e-5
        p(i) = -p(i);
    end
    
    if abs(abs(p(i))-1) < 1e-5
        nOscillation = nOscillation + 1;
        indOscillation(nOscillation) = i; 
    else
        nDamped = nDamped + 1;
        indDamped(nDamped) = i; 
    end
end
if nOscillation == n % pure oscillation
    N = 5*max(2*pi./abs(angle(p)));
elseif nOscillation == 0 % no oscillation
    [maxp,maxind] = max(abs(p));
    N = mltplcty(p,maxind)*log10(tol)/log10(maxp) + delay;
else % some of both
    pOscillation = p(indOscillation(1:nOscillation));
    pDamped      = p(indDamped(1:nDamped));
    periods = 5*max(2*pi./abs(angle(pOscillation)));
    [maxp,maxind] = max(abs(pDamped));
    N = max(periods, ...
        mltplcty(pDamped,maxind)*log10(tol)/log10(maxp)) + delay;
end
%-------------------------------------------------------------------------
function m = mltplcty( p, ind, tol)
%MLTPLCTY  Multiplicity of a pole
%   MLTPLCTY(P,IND,TOL) finds the multiplicity of P(IND) in the vector P
%   with a tolerance of TOL.  TOL defaults to .001.
if nargin < 3
    tol = .001;
end

 if any(p == 0)
   thresh = cast(tol,class(p));
 else
   thresh = tol*abs(p(ind));
 end
 
 m = 0;
for i = 1:length(p)
    if abs(p(i) - p(ind)) < thresh
        m = m + 1;
    end
end
%--------------------------------------------------------------------------
function len = lclsosimpzlength(sos,tol)

% Initialize length
firlen=1;
iirlen=1;

% Convert the filter to a transfer function.
for k=1:size(sos,1)
    
    % Get the transfer function coefficients
    b=sos(k,1:3);
    a=sos(k,4:6);
    
    if signal.internal.isfir(b,a)
        % Add the length of each FIR section
        firlen = firlen + length(b) - 1;
    else
        % Keep the maximum length of all IIR sections
        iirlen = max(iirlen, impzlength(b,a,tol));
    end
end

% Use the longest of FIR or IIR
len = max(firlen,iirlen);
```

这个函数的功能是生成滤波器的冲激响应长度，输入参数为滤波器的b，a值。请根据这个函数完成Python语言impzlength函数的编写，便于类似应用调用。

### LLM 生成 ： Python impzlength 函数

这是该函数的定义：

    import numpy as np
    from scipy import signal
    from scipy.signal import ellip, zpk2sos, firwin, butter
    
    def impzlength(b, *args):
        if isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[1] == 6:
            # SOS 矩阵情况
           # if not np.isclose(np.linalg.norm(b), 0.0):
           #     raise ValueError("参数 'b' 必须是非空的 SOS 矩阵.")
    
            if len(args) > 0:
                tol = args[0]
            else:
                tol = 0.00005
            return lclsosimpzlength(b, tol)
        else:
            # 传递函数情况
            isTF = True
            if len(args) == 0:
                a = np.array([1.0])  # 传递函数的默认 'a'
            else:
                a = args[0]
    
            if len(args) < 2:
                tol = 0.00005
            else:
                tol = float(args[1])
    
            return calculate_impz_length(b, a, tol, isTF)
    
    def calculate_impz_length(b, a, tol, isTF):
        if isTF:
            # 判断滤波器是否为 FIR（所有 'a' 的元素接近于零）
            if np.allclose(a, 0.0, atol=1e-15):
                return len(b)  # FIR 滤波器的长度
            else:
                # 找到向量 b 中第一个非零元素的索引，作为延迟 delay
                delay = find_first_nonzero_index(b)
    
                # 根据稳定性和延迟计算长度
                p = np.roots(a)
                if np.any(np.abs(p) > 1.0001):
                    N = unstable_length(p)
                else:
                    N = stableNmarginal_length(p, tol, delay)
    
                return int(np.ceil(N))-1  # 确保 N 是整数并向上取整
        else:
            return lclsosimpzlength(b, tol)
    
    def find_first_nonzero_index(arr):
        # 找到数组 arr 中第一个非零元素的索引
        indices = np.nonzero(arr)[0]  # 找到所有非零元素的索引
        if len(indices) == 0:
            return 0  # 如果数组中没有非零元素，返回索引 0（默认延迟为 0）
        else:
            return indices[0]  # 返回第一个非零元素的索引
    
    def unstable_length(p):
        ind = np.abs(p) > 1
        return int(6 / np.log10(np.max(np.abs(p[ind]))))
    
    def stableNmarginal_length(p, tol, delay):
        n = len(p)
        nOscillation = 0
        indOscillation = []
        nDamped = 0
        indDamped = []
    
        # 遍历极点 p
        for i in range(n):
            # 如果极点接近于1，则取其相反数
            if np.abs(p[i] - 1) < 1e-5:
                p[i] = -p[i]
            
            # 判断极点是振荡还是非振荡
            if np.abs(np.abs(p[i]) - 1) < 1e-5:
                nOscillation += 1
                indOscillation.append(i)
            else:
                nDamped += 1
                indDamped.append(i)
    
        if nOscillation == n:  # 全部为振荡极点
            periods = 5 * np.max(2 * np.pi / np.abs(np.angle(p)))
            N = periods
        elif nOscillation == 0:  # 全部为非振荡极点
            maxp = np.max(np.abs(p))
            maxind = np.argmax(np.abs(p))
            multiplicity = mltplcty(p, maxind, tol)
            N = multiplicity * np.log10(tol) / np.log10(maxp) + delay
        else:  # 部分振荡部分非振荡
            pOscillation = p[indOscillation]
            pDamped = p[indDamped]
            periods = 5 * np.max(2 * np.pi / np.abs(np.angle(pOscillation)))
            maxp = np.max(np.abs(pDamped))
            maxind = np.argmax(np.abs(pDamped))
            multiplicity = mltplcty(pDamped, maxind, tol)
            N = max(periods, multiplicity * np.log10(tol) / np.log10(maxp)) + delay
    
        return N
    
    def mltplcty(p, ind, tol=0.001):
        if np.any(p == 0):
            thresh = np.float(tol)
        else:
            thresh = tol * np.abs(p[ind])
        
        m = 0
        for i in range(len(p)):
            if np.abs(p[i] - p[ind]) < thresh:
                m += 1
        
        return m
    
    def lclsosimpzlength(sos, tol):
        firlen = 1
        iirlen = 1
        num_sections = sos.shape[0]
        for k in range(num_sections):
            b = sos[k, 0:3]
            a = sos[k, 3:6]
            if np.allclose(a, 0.0, atol=1e-15):
                return len(b)  # FIR 滤波器的长度
            else:
                iirlen = max(iirlen, impzlength(b, a, tol))
        return max(firlen, iirlen)


## Prompt 2 ： 应用生成的 Python impzlength.py函数做MATLAB示例对应

这是MATLAB程序1，要求为创建一个极点为 0.9 的低通全极点 IIR 滤波器。计算有效脉冲响应长度。获取脉冲响应。绘制结果图。

```
b = 1;
a = [1 -0.9];
len = impzlength(b,a)
[h,t] = impz(b,a);
stem(t,h)
```

这是impzlength函数定义

    import numpy as np
    from scipy import signal
    from scipy.signal import ellip, zpk2sos, firwin, butter
    
    def impzlength(b, *args):
        if isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[1] == 6:
            # SOS 矩阵情况
           # if not np.isclose(np.linalg.norm(b), 0.0):
           #     raise ValueError("参数 'b' 必须是非空的 SOS 矩阵.")
    
            if len(args) > 0:
                tol = args[0]
            else:
                tol = 0.00005
            return lclsosimpzlength(b, tol)
        else:
            # 传递函数情况
            isTF = True
            if len(args) == 0:
                a = np.array([1.0])  # 传递函数的默认 'a'
            else:
                a = args[0]
    
            if len(args) < 2:
                tol = 0.00005
            else:
                tol = float(args[1])
    
            return calculate_impz_length(b, a, tol, isTF)
    
    def calculate_impz_length(b, a, tol, isTF):
        if isTF:
            # 判断滤波器是否为 FIR（所有 'a' 的元素接近于零）
            if np.allclose(a, 0.0, atol=1e-15):
                return len(b)  # FIR 滤波器的长度
            else:
                # 找到向量 b 中第一个非零元素的索引，作为延迟 delay
                delay = find_first_nonzero_index(b)
    
                # 根据稳定性和延迟计算长度
                p = np.roots(a)
                if np.any(np.abs(p) > 1.0001):
                    N = unstable_length(p)
                else:
                    N = stableNmarginal_length(p, tol, delay)
    
                return int(np.ceil(N))-1  # 确保 N 是整数并向上取整
        else:
            return lclsosimpzlength(b, tol)
    
    def find_first_nonzero_index(arr):
        # 找到数组 arr 中第一个非零元素的索引
        indices = np.nonzero(arr)[0]  # 找到所有非零元素的索引
        if len(indices) == 0:
            return 0  # 如果数组中没有非零元素，返回索引 0（默认延迟为 0）
        else:
            return indices[0]  # 返回第一个非零元素的索引
    
    def unstable_length(p):
        ind = np.abs(p) > 1
        return int(6 / np.log10(np.max(np.abs(p[ind]))))
    
    def stableNmarginal_length(p, tol, delay):
        n = len(p)
        nOscillation = 0
        indOscillation = []
        nDamped = 0
        indDamped = []
    
        # 遍历极点 p
        for i in range(n):
            # 如果极点接近于1，则取其相反数
            if np.abs(p[i] - 1) < 1e-5:
                p[i] = -p[i]
            
            # 判断极点是振荡还是非振荡
            if np.abs(np.abs(p[i]) - 1) < 1e-5:
                nOscillation += 1
                indOscillation.append(i)
            else:
                nDamped += 1
                indDamped.append(i)
    
        if nOscillation == n:  # 全部为振荡极点
            periods = 5 * np.max(2 * np.pi / np.abs(np.angle(p)))
            N = periods
        elif nOscillation == 0:  # 全部为非振荡极点
            maxp = np.max(np.abs(p))
            maxind = np.argmax(np.abs(p))
            multiplicity = mltplcty(p, maxind, tol)
            N = multiplicity * np.log10(tol) / np.log10(maxp) + delay
        else:  # 部分振荡部分非振荡
            pOscillation = p[indOscillation]
            pDamped = p[indDamped]
            periods = 5 * np.max(2 * np.pi / np.abs(np.angle(pOscillation)))
            maxp = np.max(np.abs(pDamped))
            maxind = np.argmax(np.abs(pDamped))
            multiplicity = mltplcty(pDamped, maxind, tol)
            N = max(periods, multiplicity * np.log10(tol) / np.log10(maxp)) + delay
    
        return N
    
    def mltplcty(p, ind, tol=0.001):
        if np.any(p == 0):
            thresh = np.float(tol)
        else:
            thresh = tol * np.abs(p[ind])
        
        m = 0
        for i in range(len(p)):
            if np.abs(p[i] - p[ind]) < thresh:
                m += 1
        
        return m
    
    def lclsosimpzlength(sos, tol):
        firlen = 1
        iirlen = 1
        num_sections = sos.shape[0]
        for k in range(num_sections):
            b = sos[k, 0:3]
            a = sos[k, 3:6]
            if np.allclose(a, 0.0, atol=1e-15):
                return len(b)  # FIR 滤波器的长度
            else:
                iirlen = max(iirlen, impzlength(b, a, tol))
        return max(firlen, iirlen)

请应用自定义impzlength函数，实现上述MATLAB程序的功能，并给出Python程序。

### LLM 生成 ： 
```
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import ellip, zpk2sos, firwin, butter
from impzlength import impzlength
from impz import impz

# 示例1
b = np.array([1.0])
a = np.array([1.0, -0.9])
N = impzlength(b,a)
print("Impulse response length N:", N)
num_samples = 93
h, t = impz(b, a, num_samples)
# 绘制脉冲响应图
plt.stem(t,h)
plt.title('Impulse Response ')
plt.xlabel('n (samples)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```

## Prompt 3 ： 应用生成的 Python impzlength.py 函数做MATLAB示例对应

这是MATLAB程序2，要求设计了一个截止频率为0.4πrad/sample的四阶低通椭圆滤波器。指定1dB的通带纹波和60 dB的阻带衰减。将滤波器设计为极点零增益形式，并使用zp2sos获得二阶截面矩阵。根据二阶截面矩阵确定有效脉冲响应序列长度。参考下面Matlab代码，给出Python实现代码：

**Matlab代码**

```
[z,p,k] = ellip(4,1,60,.4);
[sos,g] = zp2sos(z,p,k);
len = impzlength(sos)
```

这是Python impzlength.py
```
import numpy as np
from scipy import signal
from scipy.signal import ellip, zpk2sos, firwin, butter

def impzlength(b, *args):
    if isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[1] == 6:
        # SOS 矩阵情况
       # if not np.isclose(np.linalg.norm(b), 0.0):
       #     raise ValueError("参数 'b' 必须是非空的 SOS 矩阵.")

        if len(args) > 0:
            tol = args[0]
        else:
            tol = 0.00005
        return lclsosimpzlength(b, tol)
    else:
        # 传递函数情况
        isTF = True
        if len(args) == 0:
            a = np.array([1.0])  # 传递函数的默认 'a'
        else:
            a = args[0]

        if len(args) < 2:
            tol = 0.00005
        else:
            tol = float(args[1])

        return calculate_impz_length(b, a, tol, isTF)

def calculate_impz_length(b, a, tol, isTF):
    if isTF:
        # 判断滤波器是否为 FIR（所有 'a' 的元素接近于零）
        if np.allclose(a, 0.0, atol=1e-15):
            return len(b)  # FIR 滤波器的长度
        else:
            # 找到向量 b 中第一个非零元素的索引，作为延迟 delay
            delay = find_first_nonzero_index(b)

            # 根据稳定性和延迟计算长度
            p = np.roots(a)
            if np.any(np.abs(p) > 1.0001):
                N = unstable_length(p)
            else:
                N = stableNmarginal_length(p, tol, delay)

            return int(np.ceil(N))-1  # 确保 N 是整数并向上取整
    else:
        return lclsosimpzlength(b, tol)

def find_first_nonzero_index(arr):
    # 找到数组 arr 中第一个非零元素的索引
    indices = np.nonzero(arr)[0]  # 找到所有非零元素的索引
    if len(indices) == 0:
        return 0  # 如果数组中没有非零元素，返回索引 0（默认延迟为 0）
    else:
        return indices[0]  # 返回第一个非零元素的索引

def unstable_length(p):
    ind = np.abs(p) > 1
    return int(6 / np.log10(np.max(np.abs(p[ind]))))

def stableNmarginal_length(p, tol, delay):
    n = len(p)
    nOscillation = 0
    indOscillation = []
    nDamped = 0
    indDamped = []

    # 遍历极点 p
    for i in range(n):
        # 如果极点接近于1，则取其相反数
        if np.abs(p[i] - 1) < 1e-5:
            p[i] = -p[i]
        
        # 判断极点是振荡还是非振荡
        if np.abs(np.abs(p[i]) - 1) < 1e-5:
            nOscillation += 1
            indOscillation.append(i)
        else:
            nDamped += 1
            indDamped.append(i)

    if nOscillation == n:  # 全部为振荡极点
        periods = 5 * np.max(2 * np.pi / np.abs(np.angle(p)))
        N = periods
    elif nOscillation == 0:  # 全部为非振荡极点
        maxp = np.max(np.abs(p))
        maxind = np.argmax(np.abs(p))
        multiplicity = mltplcty(p, maxind, tol)
        N = multiplicity * np.log10(tol) / np.log10(maxp) + delay
    else:  # 部分振荡部分非振荡
        pOscillation = p[indOscillation]
        pDamped = p[indDamped]
        periods = 5 * np.max(2 * np.pi / np.abs(np.angle(pOscillation)))
        maxp = np.max(np.abs(pDamped))
        maxind = np.argmax(np.abs(pDamped))
        multiplicity = mltplcty(pDamped, maxind, tol)
        N = max(periods, multiplicity * np.log10(tol) / np.log10(maxp)) + delay

    return N

def mltplcty(p, ind, tol=0.001):
    if np.any(p == 0):
        thresh = np.float(tol)
    else:
        thresh = tol * np.abs(p[ind])
    
    m = 0
    for i in range(len(p)):
        if np.abs(p[i] - p[ind]) < thresh:
            m += 1
    
    return m

def lclsosimpzlength(sos, tol):
    firlen = 1
    iirlen = 1
    num_sections = sos.shape[0]
    for k in range(num_sections):
        b = sos[k, 0:3]
        a = sos[k, 3:6]
        if np.allclose(a, 0.0, atol=1e-15):
            return len(b)  # FIR 滤波器的长度
        else:
            iirlen = max(iirlen, impzlength(b, a, tol))
    return max(firlen, iirlen)
```


### LLM 生成 ： 

下面这是调用程序
```
import numpy as np
from scipy import signal
from scipy.signal import ellip, zpk2sos, firwin, butter
from impzlength import impzlength

# 示例2
# 设计椭圆低通滤波器
N = 4  # 滤波器阶数
rp = 1  # 通带最大允许波纹（dB）
rs = 60  # 阻带最小衰减（dB）
cutoff = 0.4  # 截止频率（归一化频率，范围为0到1）
# 设计椭圆滤波器
b, a = ellip(N, rp, rs, cutoff, output='ba')
# 获取零极点增益（ZPK）表示
z, p, k = ellip(N, rp, rs, cutoff, output='zpk')
# 将零极点表示转换为二阶段（Second-Order Sections，SOS）表示
sos = zpk2sos(z, p, k)
N = impzlength(sos)
print("Impulse response length N:", N)
```



