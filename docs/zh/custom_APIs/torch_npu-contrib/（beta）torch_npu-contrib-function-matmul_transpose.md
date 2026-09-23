# （beta）torch_npu.contrib.function.matmul_transpose

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

使用NPU自定义算子替换原生写法，以提高性能。

## 函数原型

```python
torch_npu.contrib.function.matmul_transpose(tensor1, tensor2)
```

## 参数说明

- **tensor1** (`Tensor`)：第一个要乘的张量。
- **tensor2** (`Tensor`)：第二个要乘的张量。

## 返回值说明

`Tensor`

代表输出张量。

## 约束说明

在动态shape场景中，由于算子限制，不支持Box transformation deltas。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> from torch_npu.contrib.function import matmul_transpose
>>> tensor1 = torch.randn(68, 5, 75, 16).npu()
>>> tensor1.requires_grad = True
>>> tensor2 = torch.randn(68, 5, 75, 16).npu()
>>> tensor2.requires_grad = True
>>> output = matmul_transpose(tensor1, tensor2)
>>> output.sum().backward()
>>> print(output.shape)
torch.Size([68, 5, 75, 75])
```
