# （beta）torch_npu.contrib.function.matmul_transpose

## 函数原型

```
torch_npu.contrib.function.matmul_transpose(tensor1, tensor2)
```

## 功能说明

使用NPU自定义算子替换原生写法，以提高性能。

## 参数说明

- tensor1 (Tensor) - 第一个要乘的张量。
- tensor2 (Tensor) - 第二个要乘的张量。

## 输出说明

Tensor - 输出张量。

## 约束说明

在动态shape场景中，由于算子限制，不支持Box transformation deltas。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.function import matmul_transpose
>>> tensor1 = torch.randn(68, 5, 75, 16).npu()
>>> tensor1.requires_grad = True
>>> tensor2 = torch.randn(68, 5, 75, 16).npu()
>>> tensor2.requires_grad = True
>>> output = matmul_transpose(tensor1, tensor2)
>>> output.sum().backward()
```

