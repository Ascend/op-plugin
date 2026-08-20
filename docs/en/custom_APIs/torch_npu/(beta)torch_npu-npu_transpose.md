# (beta) torch_npu.npu_transpose

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. Use `torch.permute` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Returns a view of the original tensor with its dimensions permuted, and the result is contiguous. FakeTensor mode is supported.

## Prototype

```python
torch_npu.npu_transpose(self, perm, require_contiguous=True) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Input tensor.
- **`perm`** (`List[int]`): Corresponding dimension permutation.
- **`require_contiguous`** (`bool`): Specifies whether the user needs to convert the input tensor to a contiguous tensor before calling the function. The default value is `True`. When set to `False`, there is no need to convert the input tensor to a contiguous tensor before calling the function. It can be set to `True` only when the input tensor is a contiguous or a transposed tensor.

## Example

```python
>>> x = torch.randn(2, 3, 5).npu()
>>> print(x.shape)
torch.Size([2, 3, 5])
>>> x1 = torch_npu.npu_transpose(x, (2, 0, 1))
>>> print(x1.shape)
torch.Size([5, 2, 3])
>>> x2 = torch_npu.npu_transpose(x, (2, 0, 1))
>>> print(x2.shape)
torch.Size([5, 2, 3])
```
