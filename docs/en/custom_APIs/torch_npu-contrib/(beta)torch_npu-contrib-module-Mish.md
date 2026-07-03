# (beta) torch_npu.contrib.module.Mish

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.nn.Mish` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Applies an NPU-based Mish operation.

## Prototype

```python
torch_npu.contrib.module.Mish(nn.Module)
```

## Example

```python
>>> import torch
>>> from torch_npu.contrib.module import Mish
>>> m = Mish()
>>> input_tensor = torch.randn(2, 32, 5, 5).npu()
>>> output = m(input_tensor)
```
