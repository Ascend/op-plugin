# (beta) torch_npu.npu_get_float_status

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

Obtains the overflow detection result.

## Prototype

```python
torch_npu.npu_get_float_status(self) -> Tensor
```

## Parameters

**`self`** (`Tensor`): A tensor containing the data memory address. The data type is `torch.float32`.

## Return Values

`Tensor`

Returns the detection result.

## Examples

```python
>>> import torch, torch_npu
>>> x = torch.rand(2).npu()
>>> torch_npu.npu_get_float_status(x)
tensor([0, 0, 0, 0, 0, 0, 0, 0], device='npu:0', dtype=torch.int32)
```
