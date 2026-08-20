# (beta) torch_npu.npu_pad

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. Use `torch.nn.functional.pad` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Pads a tensor.

## Prototype

```python
torch_npu.npu_pad(input, paddings) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Input tensor.
- **`paddings`** (`List[int]`): The data type can be `int32` or `int64`.

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
