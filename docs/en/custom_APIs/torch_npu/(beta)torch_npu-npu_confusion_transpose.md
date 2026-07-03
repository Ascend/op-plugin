# (beta) torch_npu.npu_confusion_transpose

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.view` and `torch.permute` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Obfuscates the reshape and transpose operations.

## Prototype

```python
torch_npu.npu_confusion_transpose(self, perm, shape, transpose_first) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Input tensor. The data type can be `float16`, `float32`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, or `uint64`.
- **`perm`** (`List[int]`): Dimension permutation of the `self` tensor.
- **`shape`** (`List[int]`): Input shape.
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
