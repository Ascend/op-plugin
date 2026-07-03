# (beta) torch_npu.contrib.module.SiLU

> [!NOTICE]  
>This API is planned for deprecation. Use `torch.nn.SiLU` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Applies the NPU-based Sigmoid Linear Unit (SiLU) function element-wise. The SiLU function is also known as the Swish function.

## Prototype

```python
torch_npu.contrib.module.SiLU(nn.Module)
```

## Example

```python
>>> import torch
>>> import torch_npu
>>> from torch_npu.contrib.module import SiLU
>>> m = SiLU()
>>> input_tensor = torch.randn(2, 32, 5, 5).npu()
>>> output = m(input_tensor)
```
