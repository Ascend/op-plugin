# torch_npu.npu_gather_sparse_index

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |

## 函数原型


```
torch_npu.npu_gather_sparse_index(input, index) -> torch.Tensor
```

## 功能说明: 

从输入Tensor的指定维度，按照`index`中的下标序号提取元素，保存到输出Tensor中。

示例：

输入x如下：
$$
\begin{bmatrix}
  1& 2 & 3\\
  4& 5 & 6\\
  7& 8 &9
\end{bmatrix}
$$
索引idx如下：
$$
\begin{bmatrix}
 1 & 0
\end{bmatrix}
$$
此时输入shape为$[3, 3]$，索引shape为$[2]$，则输出shape为$[2, 3]$，输出结果如下：
$$
\begin{bmatrix}
 4 & 5 & 6\\
 7 & 8 &9
\end{bmatrix}
$$


## 参数说明

**input**(`Tensor`)：输入张量，数据维度仅支持2维。数据类型支持`float32`, `float16`, `bfloat16`, `int64`, `int32`, `int16`，`int8`, `uint8`, `bool`, `float64`, `complex64`, `complex128`。

**index**(`Tensor`)：包含目标元素下标序号的张量。数据维度不超过7维。数据类型支持`int64`, `int32`。取值范围$[0, input.shape[0] - 1]$, 不支持负数索引。

## 输出说明

`Tensor`
接口计算获得的结果，包含按照`index`中的下标序号提取的元素。数据类型与`input`一致，输出维度为$index.dim + input.dim - 1$。例如`input.shape = [16, 32]`, `index.shape = [2, 3]`，则输出张量 `out.shape = [2, 3, 32]`。

## 约束说明
- `input`的维度与`index`的维度之和减1不能超过8，即$index.dim + input.dim - 1<=8$。
- 为获取性能收益，`weight`和`index`需要满足如下约束：
     1. `weight`的shape内积需要大于$150 * 1024 / itemsize$，其中itemsize为`weight` dtype对应元素大小，可以通过`torch.dtype.itemsize`查询。
     2. `index`的shape内积大于960。
     3. 数据需要聚合，即非0值分布集中，0分布集中。


## 调用示例

```python
import torch
import torch_npu

inputs = torch.randn(16, 32).npu()
index = torch.randint(0, 16, [2, 3]).npu()
out = torch_npu.npu_gather_sparse_index(inputs, index)
```