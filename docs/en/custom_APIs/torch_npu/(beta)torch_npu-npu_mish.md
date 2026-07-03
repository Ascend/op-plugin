# (beta) torch_npu.npu_mish

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.nn.functional.mish` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Computes the Mish activation function element-wise. The Mish activation function is defined as: `$mish(x) = x * tanh(softplus(x))$, where $softplus(x) = ln(1 + e^x)$.

## Prototype

```python
torch_npu.npu_mish(self) -> Tensor
```

## Parameters

**`self`** (`Tensor`): The data type can be `float16` or `float32`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> x = torch.rand(10, 30, 10).npu()
>>> y = torch_npu.npu_mish(x)
>>> print(y.shape)
torch.Size([10, 30, 10])
```
