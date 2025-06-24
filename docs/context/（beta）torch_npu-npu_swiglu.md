# （beta）torch\_npu.npu\_swiglu

## 函数原型

```
torch_npu.npu_swiglu(Tensor input, int dim=-1) -> (Tensor)
```

## 功能说明

Swish门控线性单元激活函数，实现张量input的SwiGlu计算。

公式如下：

![](./figures/zh-cn_formulaimage_0000002284552989.png)

-   “x“是输入参数input的Tensor。
-   “dim“是切分维度，默认为-1。
-   “A“和“B“是x沿dim维度切分的Tensor。A表示前半部分张量，B表示后半部分张量。

## 参数说明

**input**：Tensor类型，表示待计算的数据，shape支持1-8维，且shape必须在入参dim对应维度上可以整除2。不支持非连续的Tensor，不支持空Tensor。dtype支持fp32、fp16或bf16类型。

**dim**：Int类型，默认为-1。需要进行切分的维度序号，对input相应轴进行对半切。取值范围为\[-input.dim\(\), input.dim\(\)-1\]。

## 输出说明

输出为Tensor，计算公式的最终输出outputs。数据类型与计算输入input的类型一致，不支持非连续的Tensor。

## 支持的型号

<term>Atlas A2 训练系列产品</term>

## 调用示例

```python
import torch
import torch_npu
input_tensor = torch.randn(2, 32, 6, 6)
output = torch_npu.npu_swiglu(input_tensor.npu(), dim = -1)
```

