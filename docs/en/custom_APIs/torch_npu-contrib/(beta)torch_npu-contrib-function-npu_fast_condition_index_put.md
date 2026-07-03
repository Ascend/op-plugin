# (beta) torch_npu.contrib.function.npu_fast_condition_index_put

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Replaces the native Boolean `index_put` implementation with an NPU-optimized implementation.

## Prototype

```python
torch_npu.contrib.function.npu_fast_condition_index_put(x, condition, value)
```

## Parameters

- **`x`** (`Tensor`): Input tensor.
- **`condition`** (`BoolTensor`): Condition tensor.
- **`value`** (`int` or `float`): Value to assign to the selected bounding box elements.

## Return Values

`Tensor`

Bounding box transformation deltas.

## Example

```python
>>> import torch
>>> from torch_npu.contrib.function import npu_fast_condition_index_put
>>> import copy
>>> x = torch.randn(128, 8192).npu()
>>> condition = x < 0.5
>>> value = 0.
>>> x1 = copy.deepcopy(x)[condition] = value
>>> x1_opt = npu_fast_condition_index_put(x, condition, value)
>>> print(x1_opt)
tensor([[0.9661, 1.6750, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [1.3621, 0.0000, 0.9606,  ..., 1.4324, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0000, 0.0000,  ..., 1.4605, 0.7734, 1.9721],
        [0.0000, 0.9325, 0.7112,  ..., 0.0000, 0.9814, 1.4227],
        [1.0037, 0.0000, 0.0000,  ..., 0.0000, 1.6497, 0.0000]],
       device='npu:0')
>>> print(x1_opt.shape)
torch.Size([128, 8192])

```
