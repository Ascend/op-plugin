# (beta) torch_npu.npu_confusion_transpose

> [!NOTICE]  
> This API is planned for deprecation. Use the `Tensor.view()` and `torch.permute` APIs instead.

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->

## Function

Fuses the reshape and transpose operations.

## Prototype

```python
torch_npu.npu_confusion_transpose(self, perm, shape, transpose_first) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Input tensor. The data type can be `torch.float16`, `torch.float32`, `torch.int8`, `torch.int16`, `torch.int32`, `torch.int64`, `torch.uint8`, `torch.uint16`, `torch.uint32`, or `torch.uint64`.
- **`perm`** (`List[int]`): Dimension permutation of the `self` tensor.
- **`shape`** (`List[int]`): The target shape after the reshape operation.
- **`transpose_first`** (`bool`): If set to `True`, transpose is executed first. Otherwise, reshape is executed first.

## Example

```python
>>> x = torch.rand(2, 3, 4, 6).npu()
>>> print(x.shape)
torch.Size([2, 3, 4, 6])
>>> y = torch_npu.npu_confusion_transpose(x, (0, 2, 1, 3), (2, 4, 18), True)
>>> print(y.shape)
torch.Size([2, 4, 18])
>>> y2 = torch_npu.npu_confusion_transpose(x, (0, 2, 1), (2, 12, 6), False)
>>> print(y2.shape)
torch.Size([2, 6, 12])
```
