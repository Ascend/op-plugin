# (beta) torch_npu.npu_sign_bits_pack

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

- Description: Packs `float` type inputs into `uint8` types. Every 8 floating-point numbers are packed into one `uint8` value, `-1.0` is encoded as binary bit `0`, and `1.0` is encoded as binary bit `1`, and packed in little-endian order.

- Equivalent computation logic:
    
     `sign_pack` can be used as an equivalent replacement for `torch_npu.npu_sign_bits_pack`. The computation logic of the two operators is identical.
     
     ```python
    import torch
    import numpy as np
    
    def sign_pack(in_data, size):
        sign_data = np.sign(in_data)
        sign_data = sign_data + 1
        bool_data = np.bool_(sign_data)
        pack_bit = np.packbits(bool_data, bitorder="little")
        return pack_bit.reshape(size, pack_bit.shape[0] // size)
    ```

## Prototype

```python
torch_npu.npu_sign_bits_pack(self, size) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. 1D float tensor. The data type can be `float32` and `float16`.
- **`size`** (`int`): Required. Used to reshape the first dimension of the output tensor.

## Return Values

`Tensor`

Packed tensor.

## Constraints

`size` can be divided by the packed `float` output. If the size of `self` is divisible by 8, the size of the output is $(\text{size of }self)/8$. Otherwise, the size of the output is $(\text{size of }self // 8) + 1$. A `-1` floating-point value will be added at the little-endian position to fill divisibility.

## Example

```python
>>> import torch, torch_npu
>>> a = torch.tensor([5,4,3,2,0,-1,-2, 4,3,2,1,0,-1,-2],dtype=torch.float32).npu()
>>> b = torch_npu.npu_sign_bits_pack(a, 2)
>>> print(b)
tensor([[159],
        [ 15]], device='npu:0', dtype=torch.uint8)
>>> c = torch_npu.npu_sign_bits_pack(a, 1)
>>> print(c)
tensor([[159, 15]], device='npu:0', dtype=torch.uint8)
```
