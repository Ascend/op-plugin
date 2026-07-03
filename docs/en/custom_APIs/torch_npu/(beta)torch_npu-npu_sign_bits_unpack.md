# (beta) torch_npu.npu_sign_bits_unpack

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

- Description: Unpacks a `uint8` type input into a `float` type output. Each `uint8` value is decoded into eight floating-point numbers from its eight binary bits, where `0` is decoded as `-1.0`, and `1` is decoded as `1.0`. The decoded values are returned in little-endian order.

- Equivalent computation logic:
    
     `sign_unpack` can be used as an equivalent replacement for `torch_npu.npu_sign_bits_unpack`. The computation logic of the two operators is identical.
     
     ```python
    import torch
    import numpy as np
    
    def sign_unpack(in_data, size, dtype):
        unpack_data = np.unpackbits(in_data, bitorder="little")
        unpack_data = unpack_data.astype(dtype)
        unpack_data = (unpack_data - 0.5) * 2.0
        return unpack_data.reshape(size, unpack_data.shape[0] // size)
    ```

## Prototype

```python
torch_npu.npu_sign_bits_unpack(x, size, dtype) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. 1D `uint8` tensor.
- **`size`** (`int`): Required. Used to reshape the first dimension of the output tensor.
- **`dtype`** (`torch.dtype`): Required. If set to `torch.float16`, the output data type is `float16`. If set to `torch.float32`, the output data type is `float32`.

## Return Values

`Tensor`

Unpacked tensor.

## Constraints

`size` must be divisible by the unpacked `uint8` output size of `x`. The unpacked `uint8` output size of `x` is $(\text{size of } x) * 8$.

## Example

```python
>>> import torch, torch_npu
>>> a = torch.tensor([159, 15], dtype=torch.uint8).npu()
>>> b = torch_npu.npu_sign_bits_unpack(a, 2, torch.float32)
>>> print(b)
tensor([[ 1.,  1.,  1.,  1.,  1., -1., -1.,  1.],
        [ 1.,  1.,  1.,  1., -1., -1., -1., -1.]], device='npu:0')
>>> c = torch_npu.npu_sign_bits_unpack(a, 1, torch.float32)
>>> print(c)
tensor([[ 1.,  1.,  1.,  1.,  1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1., -1.]], device='npu:0')
```
