# (beta) torch_npu.npu_sort_v2

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.sort` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Sorts the elements of the input tensor in ascending order along the specified dimension without returning indices. If `dim` is not specified, the last dimension of the input is selected. If `descending` is set to `True`, the elements are sorted in descending order by value.

## Prototype

```python
torch_npu.npu_sort_v2(self, dim=-1, descending=False, out=None) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Input tensor.
- **`dim`** (`int`): Optional. Dimension along which to sort. The default value is `-1`.
- **`descending`** (`bool`): Optional. Sort order control (ascending or descending). The default value is `False`.

## Constraints

Currently, only the last dimension of the input (`dim=-1`) is supported.

## Example

```python
>>> import torch
>>> import torch_npu
>>> x = torch.randn(3, 4).npu()
>>> print(x)
tensor([[-0.0067,  1.7790,  0.5031, -1.7217],
        [ 1.1685, -1.0486, -0.2938,  1.3241],
        [ 0.1880, -2.7447,  1.3976,  0.7380]], device='npu:0')
>>> sorted_x = torch_npu.npu_sort_v2(x)
>>> print(sorted_x)
tensor([[-1.7217, -0.0067,  0.5029,  1.7793],
        [-1.0488, -0.2937,  1.1689,  1.3242],
        [-2.7441,  0.1880,  0.7378,  1.3975]], device='npu:0')
```
