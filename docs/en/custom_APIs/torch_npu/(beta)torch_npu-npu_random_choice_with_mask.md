# (beta) torch_npu.npu_random_choice_with_mask

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Obtains the indices of non-zero elements and outputs the shuffled results.

## Prototype

```python
torch_npu.npu_random_choice_with_mask(x, count=256, seed=0, seed2=0) -> (Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input tensor. Only the `bool` data type is supported.
- **`count`** (`int`): Optional. Output count. The default value is `256`. If set to `0`, all non-zero elements are output.
- **`seed`** (`int`): Optional. Random seed value. The default value is `0`. The data type can be `int32` or `int64`.
- **`seed2`** (`int`): Optional. Random seed value. The default value is `0`. The data type can be `int32` or `int64`.

## Return Values

- **`y`** (`Tensor`): Indices of non-zero elements. This parameter must be a 2D tensor.
- **`mask`** (`Tensor`): Determines whether the corresponding index is valid. This parameter must be a 1D tensor.

## Example

```python
>>> import torch, torch_npu
>>> x = torch.tensor([1, 0, 1, 0], dtype=torch.bool).to("npu")
>>> result, mask = torch_npu.npu_random_choice_with_mask(x, 2, 1, 0)
>>> print(result)
tensor([[0],[2]], device='npu:0', dtype=torch.int32)
>>> print(mask)
tensor([True, True], device='npu:0')
```
