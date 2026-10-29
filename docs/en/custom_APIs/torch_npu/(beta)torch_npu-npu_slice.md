# (beta) torch_npu.npu_slice

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

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
- **`offsets`** (`List[int]`): The data type can be `torch.int32` or `torch.int64`.
- **`size`** (`List[int]`): The data type can be `torch.int32` or `torch.int64`.

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
