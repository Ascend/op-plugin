# (beta) torch_npu.npu_indexing

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

Slices the input tensor by using `begin` as the start index, `end` as the end index, and `strides` as the stride.

## Prototype

```python
torch_npu.npu_indexing(self, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. Input tensor.
- **`begin`** (`List[int]`): Required. Index of the first value to be selected.
- **`end`** (`List[int]`): Required. Index of the last value to be selected.
- **`strides`** (`List[int]`): Required. Index increments.
- **`begin_mask`** (`int`): Optional. Bitmask, where bit `i` being `1` indicates that the start value is ignored and the maximum possible interval is used. The default value is `0`.
- **`end_mask`** (`int`): Optional. Similar to `begin_mask`. The default value is `0`.
- **`ellipsis_mask`** (`int`): Optional. Bitmask, where bit `i` being `1` indicates that the `i`-th position is actually an ellipsis. The default value is `0`.
- **`new_axis_mask`** (`int`): Optional. Bitmask, where bit `i` being `1` indicates that a new 1D shape is created at the `i`-th position. The default value is `0`.
- **`shrink_axis_mask`** (`int`): Optional. Bitmask, where bit `i` being `1` indicates that the dimension at position `i` should be shrunk. The default value is `0`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> input = torch.tensor([[1, 2, 3, 4],[5, 6, 7, 8]], dtype=torch.int32).to("npu")
>>> print(input)
tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]], device='npu:0', dtype=torch.int32)
>>> output = torch_npu.npu_indexing(input, [0, 0], [2, 2], [1, 1])
>>> print(output)
tensor([[1, 2],
        [5, 6]], device='npu:0', dtype=torch.int32)
```
