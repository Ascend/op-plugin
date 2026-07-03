# (beta) torch_npu.npu_alloc_float_status

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Allocates a tensor dedicated to storing floating-point operation status flags. This tensor is used to record overflow status during subsequent computations.

## Prototype

```python
torch_npu.npu_alloc_float_status(input) -> Tensor
```

## Parameters

**`input`** (`Tensor`): Required. An arbitrary NPU tensor, mainly used to determine the device information.

## Return Values

`Tensor`

A tensor containing eight `float32` zero values.

## Example

```python
>>> import torch
>>> import torch_npu
>>> input = torch.randn([1,2,3]).npu()
## Allocate status space
>>> output = torch_npu.npu_alloc_float_status(input)
>>> print(input)
tensor([[[ 2.2324,  0.2478, -0.1056],
        [ 1.1273, -0.2573,  1.0558]]], device='npu:0')
>>> print(output)
tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')

## Clear status
>>> torch_npu.npu_clear_float_status(output)

## Execute computation operations that may overflow
## ...forward/backward propagation...

## Obtain the detection result
>>> result = torch_npu.npu_get_float_status(output)

```
