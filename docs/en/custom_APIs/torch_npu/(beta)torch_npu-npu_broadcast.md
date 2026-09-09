# (beta) torch_npu.npu_broadcast

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.broadcast_to` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Returns a new view of `self` with singleton dimensions expanded, and the result is contiguous. The tensor can also be expanded by more dimensions, and new dimensions are added at the front.

## Prototype

```python
torch_npu.npu_broadcast(self, size) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Input tensor.
- **`size`** (`List[int]`): Corresponding expanded size.

## Example

```python
>>> import torch, torch_npu
>>> x = torch.tensor([[1],[2],[3]]).npu()
>>> print(x.shape)
torch.Size([3, 1])
>>> print(torch_npu.npu_broadcast(x, [3,4]))
tensor([[1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]], device='npu:0')
```
