# (beta) torch_npu.npu_slice

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. Use `torch.slice` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Extracts a slice from a tensor.

## Prototype

```python
torch_npu.npu_slice(self, offsets, size) -> Tensor
```

> [!CAUTION]  
> This API does not support backward computation.

## Parameters

- **`self`** (`Tensor`): Input tensor.
- **`offsets`** (`List[int]`): The data type can be `int32` or `int64`.
- **`size`** (`List[int]`): The data type can be `int32` or `int64`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> input = torch.tensor([[1,2,3,4,5], [6,7,8,9,10]], dtype=torch.float16).to("npu")
>>> offsets = [0, 0]
>>> size = [2, 2]
>>> output = torch_npu.npu_slice(input, offsets, size)
>>> print(output)
tensor([[1., 2.],
        [6., 7.]], device='npu:0', dtype=torch.float16)
```
