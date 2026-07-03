# (beta) torch_npu.npu_min

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.min` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Computes the minimum values along `dim`. This API is similar to `torch.min` and is optimized for NPUs.

## Prototype

```python
torch_npu.npu_min(self, dim, keepdim=False) -> (Tensor, Tensor)
```

## Parameters

- **`self`** (`Tensor`): Input tensor.
- **`dim`** (`int`): Dimension to be reduced.
- **`keepdim`** (`bool`): Specifies whether to retain `dim` in the output tensor.

## Return Values

- **`values`** (`Tensor`): Minimum values along the specified dimension in the input tensor.
- **`indices`** (`Tensor`): Indices of the minimum values in the input tensor.

## Example

```python
>>> import torch
>>> import torch_npu
>>> input = torch.randn(2, 2, 2, 2, dtype = torch.float32).npu()
>>> print(input)
tensor([[[[-0.9909, -0.2369],
          [-0.9569, -0.6223]],

        [[ 0.1157, -0.3147],
          [-0.7761,  0.1344]]],

        [[[ 1.6292,  0.5953],
          [ 0.6940, -0.6367]],

        [[-1.2335,  0.2131],
          [ 1.0748, -0.7046]]]], device='npu:0')
>>> outputs, indices = torch_npu.npu_min(input, 2)
>>> print(outputs)
tensor([[[-0.9909, -0.6223],
        [-0.7761, -0.3147]],

        [[ 0.6940, -0.6367],
        [-1.2335, -0.7046]]], device='npu:0')
>>> print(indices)
tensor([[[0, 1],
        [1, 0]],

        [[1, 1],
        [0, 1]]], device='npu:0', dtype=torch.int32)
```
