# (beta) torch_npu.copy_memory_

> **Notice**:<br>
>This API is planned for deprecation. Use `torch.Tensor.copy_` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Copies elements from `src` to `self` and returns `self` in place.

## Prototype

```python
torch_npu.copy_memory_(self, src, non_blocking=False) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. Destination tensor of the copy operation, which receives data.
- **`src`** (`Tensor`): Required. Source tensor of the copy operation, which provides data.
- **`non_blocking`** (`bool`): Optional. The default value is `False`. If set to `True`, the copy may occur asynchronously with respect to the host. In other cases, this parameter has no effect.

## Constraints

`copy_memory_` supports only NPU tensors. The input tensors to `copy_memory_` must have the identical data type and device index.

## Example

```python
>>> a=torch.IntTensor([0,  0, -1]).npu()
>>> b=torch.IntTensor([1, 1, 1]).npu()
>>> torch_npu.copy_memory_(a, b)
tensor([1, 1, 1], device='npu:0', dtype=torch.int32)
```
