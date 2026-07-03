# (beta) torch_npu.npu_dtype_cast

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.to` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Converts the data type (`dtype`) of a tensor. FakeTensor mode is supported.

## Prototype

```python
torch_npu.npu_dtype_cast(input, dtype) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Input tensor.
- **`dtype`** (`torch.dtype`): Target data type of the returned tensor.

## Examples

Example 1:

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu_dtype_cast(torch.tensor([0, 0.5, -1.]).npu(), dtype=torch.int)
tensor([ 0,  0, -1], device='npu:0', dtype=torch.int32)
```

Example 2:

```python
#FakeTensor mode
>>> import torch
>>> import torch_npu
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     x = torch.rand(2, dtype=torch.float32).npu()
...     res = torch_npu.npu_dtype_cast(x, torch.float16)
...
>>> print(res)
FakeTensor(..., device='npu:0', size=(2,), dtype=torch.float16)
```
