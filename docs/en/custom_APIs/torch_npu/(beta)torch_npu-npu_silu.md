# (beta) torch_npu.npu_silu

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.nn.functional.silu` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Computes the Swish activation function of `self`. Swish is an activation function defined as $[x * \text{sigmoid}(x)]$.

## Prototype

```python
torch_npu.npu_silu(self) -> Tensor
```

## Parameters

**`self`** (`Tensor`): The data type can be `float16` or `float32`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> a=torch.rand(2,8).npu()
>>> output = torch_npu.npu_silu(a)
>>> print(output)
tensor([[0.4397, 0.7178, 0.5190, 0.2654, 0.2230, 0.2674, 0.6051, 0.3522],
        [0.4679, 0.1764, 0.6650, 0.3175, 0.0530, 0.4787, 0.5621, 0.4026]],
       device='npu:0')
```
