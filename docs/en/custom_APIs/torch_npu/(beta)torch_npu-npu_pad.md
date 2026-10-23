# (beta) torch_npu.npu_pad

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

Pads a tensor.

## Prototype

```python
torch_npu.npu_pad(input, paddings) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Input tensor.
- **`paddings`** (`List[int]`): The data type can be `torch.int32` or `torch.int64`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> input = torch.tensor([[20, 20, 10, 10]], dtype=torch.float16).to("npu")
>>> paddings = [1, 1, 1, 1]
>>> output = torch_npu.npu_pad(input, paddings)
>>> print(output)
tensor([[ 0.,  0.,  0.,  0.,  0.,  0.],
        [ 0., 20., 20., 10., 10.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.]], device='npu:0', dtype=torch.float16)
```
