# (beta) torch_npu.fast_gelu

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Computes the Fast Gaussian Error Linear Units (FastGelu) activation function for each element in the input tensor. FakeTensor mode is supported.

## Prototype

```python
torch_npu.fast_gelu(self) -> Tensor
```

## Parameters

**`self`** (`Tensor`): The data type can be `float16` or `float32`.

## Examples

Example 1:

```python
>>> import torch
>>> import torch_npu
>>> x = torch.rand(2).npu()
>>> print(x)
tensor([0.5991, 0.4094], device='npu:0')
>>> print(torch_npu.fast_gelu(x))
tensor([0.4403, 0.2733], device='npu:0')
```

Example 2:

```python
>>> import torch
>>> import torch_npu
# FakeTensor mode
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     x = torch.rand(2).npu()
...     torch_npu.fast_gelu(x)
>>> FakeTensor(..., device='npu:0', size=(2,))
```
