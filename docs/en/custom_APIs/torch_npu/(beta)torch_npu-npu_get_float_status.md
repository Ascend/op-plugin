# (beta) torch_npu.npu_get_float_status

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Obtains the overflow detection result.

## Prototype

```python
torch_npu.npu_get_float_status(self) -> Tensor
```

## Parameters

**`self`** (`Tensor`): A tensor containing the data memory address. The data type is `float32`.

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
