# (beta) torch_npu.contrib.function.roll

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Replaces the native `roll` operation in Swin Transformer with an NPU-optimized implementation.

## Prototype

```python
torch_npu.contrib.function.roll(input1, shifts, dims)
```

## Parameters

- **`input1`** (`Tensor`): Input tensor.
- **`shifts`** (`Tuple[int]`): Roll offsets for each dimension.
- **`dims`** (`Tuple[int]`): Dimensions to roll.

## Return Values

`Tensor`

Rolled output tensor.

## Constraints

**`input1`** must be a 4D tensor. The lengths of `shifts` and `dims` must be `2`.

## Example

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.function import roll
>>> input1 = torch.randn(32, 56, 56, 16).npu()
>>> shift_size = 3
>>> shifted_x_npu = roll(input1, shifts=(-shift_size, -shift_size), dims=(1, 2))
>>> print(shifted_x_npu.shape)
torch.Size([32, 56, 56, 16])
```
