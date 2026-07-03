# (beta) torch_npu.one_

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.fill_` or `torch.ones_like` instead. Note that `torch.ones_like` is a non-in-place API.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Fills the `self` tensor with `1`s.

## Prototype

```python
torch_npu.one_(self) -> Tensor
```

## Parameters

**`self`** (`Tensor`): Input tensor.

## Example

```python
>>> import torch
>>> import torch_npu
>>> x = torch.rand(2, 3).npu()

>>> print(x)
tensor([[0.8517, 0.1428, 0.0839],
        [0.1416, 0.9540, 0.9125]], device='npu:0')

>>> print(torch_npu.one_(x))
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='npu:0')
```
