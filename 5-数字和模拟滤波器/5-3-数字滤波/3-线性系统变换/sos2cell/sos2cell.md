```python
# 信号处理仿真与应用 - 数字和模拟滤波器 - 线性系统变换

## MATLAB函数描述：cell2sos 

函数来源：[MATLAB envelope](https://ww2.mathworks.cn/help/signal/ref/sos2cell.html)

### 语法

c = sos2cell(m)
c = sos2cell(m,g)

### 说明
1.c = sos2cell(m)将cell2sos生成的L行6列矩阵m变为由一行L列单元阵列组成的一行二列单元阵列c。可以使用 c 指定一个具有 L 级联二阶截面的量化滤波器。
矩阵 m 的形式应为m = [b1 a1;b2 a2; ... ;bL aL] 。其中，i=1，...，L 的 bi 和 ai 均为 一行3列向量。由此得到的 c 是一个 1×L 的单元格数组，
其形式为c = { {b1 a1} {b2 a2} ... {bL aL} }。
2.c = sos2cell(m,g)带有可选的增益项 g，会在c中预置常数g。c = {{g,1} {b1,a1} {b2,a2}...{bL,aL} }。


### 输入参数

m — 输入序列L行6列矩阵
输入序列，指定L行6列矩阵。
示例:  m = [1 2 3 3 2 1;4 5 6 6 5 4; ... ;bL aL] 。

g —可选的增益项
输出的第一个向量为{g,1}。

### 输出参量

c — 
当c = sos2cell(m)，c = { {b1 a1} {b2 a2} ... {bL aL} }。
当c = sos2cell(m,g)，c = {{g,1} {b1,a1} {b2,a2}...{bL,aL} }。
bi,ai为1乘3数组。

## Python函数描述：cell2sos

函数来源：自定义

### 包络函数定义：

def sos2cell(m, g=None):
    cell_array = []
    
    # 如果提供了增益项g，则将其添加到第一个单元格数组中
    if g is not None:
        cell_array.append([g, 1])
    
    # 遍历二阶节矩阵m，并将其转换为单元格数组
    for row in m:
        cell_array.append([row[:3], row[3:]])
    
    return cell_array


这段代码定义了一个名为 `cell2sos` 的函数，实现和matlab相似的功能，
### 参数

m：二阶节矩阵，由tf2sos生成，每行包含一个二阶节的系数，每个系数由两个1x3行向量组成，分别代表分子和分母多项式的系数。
g（可选）：增益项，作为常量值添加到结果单元格数组的开头。
返回值

### 返回值

单元格数组 c，包含了将二阶节矩阵转换为单元格数组的结果。如果提供了增益项 g，则结果的第一个单元格数组包含增益项 g 和常量值 1。

### 注意事项

请确保输入的二阶节矩阵 m 符合预期的格式，每行包含两个长度为 3 的系数向量。
增益项 g 是可选的，如果不提供，则结果单元格数组将不包含增益项。

### 函数工作原理

该函数首先检查是否提供了增益项 g，如果提供了，则将其添加到结果单元格数组的开头。
然后，它遍历二阶节矩阵 m，将每个二阶节的系数转换为单元格数组，并将这些单元格数组按顺序添加到结果数组中。

### 使用场景

当需要将数字滤波器表示从二阶节矩阵形式转换为单元格数组形式时，可以使用该函数。
用于指定具有串联二阶节的量化滤波器的参数。

### 改进建议

考虑添加输入参数的验证功能，以确保输入的二阶节矩阵格式正确。
可以增加对其他滤波器表示形式的转换支持，提高函数的通用性和灵活性。

## Prompt 1 ： 生成 Python envelope 函数

参考下面MATLAB代码的envelope函数
```
m=[1 2 3 3 2 1];
g=2

c = sos2cell(m)
m= sos2cell(m,g)

```

和我们采用Python语言实现的解析信号计算啁啾信号的上下包络，
```
def sos2cell(m, g=None):
    cell_array = []
    
    # 如果提供了增益项g，则将其添加到第一个单元格数组中
    if g is not None:
        cell_array.append([g, 1])
    
    # 遍历二阶节矩阵m，并将其转换为单元格数组
    for row in m:
        cell_array.append([row[:3], row[3:]])
    
    return cell_array

# 示例输入：二阶节矩阵m和可选的增益项g
m = [[1, 2,3, 3, 2, 1]]
g = 2

# 将二阶节矩阵转换为单元格数组
cell_array = sos2cell(m, g)

# 输出单元格数组
print("Cell array:")
for i, cell in enumerate(cell_array):
    print(f"Filter {i + 1}: {cell}")

```



```
