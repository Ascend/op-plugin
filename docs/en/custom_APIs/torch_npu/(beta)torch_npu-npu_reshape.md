# (beta) torch_npu.npu_reshape

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.reshape` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Reshapes a tensor. This operation only changes the tensor shape while its data remains unchanged.

## Prototype

```python
torch_npu.npu_reshape(self, shape, bool can_refresh=False) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Input tensor.
- **`shape`** (`List[int]`): Defines the shape of the output tensor.
- **`can_refresh`** (`bool`): Specifies whether to refresh reshape in place. The default value is `False`.

## Constraints

This operator cannot be directly called by the `aclopExecute` API.

## Example

```python
>>> import torch
>>> import torch_npu
>>> a=torch.rand(2,8).npu()
>>> out=torch_npu.npu_reshape(a,(4,4))
>>> print(out)
tensor([[0.6657, 0.9857, 0.7614, 0.4368],
        [0.3761, 0.4397, 0.8609, 0.5544],
        [0.7002, 0.3063, 0.9279, 0.5085],
        [0.1009, 0.7133, 0.8118, 0.6193]], device='npu:0')
```
