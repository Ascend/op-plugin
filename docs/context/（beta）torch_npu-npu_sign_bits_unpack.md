# （beta）torch\_npu.npu\_sign\_bits\_unpack

## 函数原型

```
torch_npu.npu_sign_bits_unpack(x, size, dtype) -> Tensor
```

## 功能说明

-   API功能：将uint8类型的输入拆包为float类型。将uint8数值中的8个二进制位解码为8个浮点数，0解码为-1.0，1解码为1.0，并以小端序进行返回。

-   小算子等价计算逻辑：
    
     可使用`sign_unpack`等价替换`torch_npu.npu_sign_bits_unpack`，两者计算逻辑一致。
     
     ```python
    import torch
    import numpy as np
    
    def sign_unpack(in_data, size, dtype):
        unpack_data = np.unpackbits(in_data, bitorder="little")
        unpack_data = unpack_data.astype(dtype)
        unpack_data = (unpack_data - 0.5) * 2.0
        return unpack_data.reshape(size, unpack_data.shape[0] // size)
    ```
## 参数说明

-   x(Tensor): 必选参数，1D uint8张量。
-   size(Int): 必选参数，用于reshape输出张量的第一个维度。
-   dtype(torch.dtype): 必选参数，值为torch.float16设置输出类型为float16，值为torch.float32设置输出类型为float32。

## 返回值说明
`Tensor`

拆包后的张量。

## 约束说明

size可被x的uint8拆包输出大小整除。x的uint8拆包输出大小为(size of x) * 8。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> import torch, torch_npu
>>> a = torch.tensor([159, 15], dtype=torch.uint8).npu()
>>> b = torch_npu.npu_sign_bits_unpack(a, 2, torch.float32)
>>> b
tensor([[ 1.,  1.,  1.,  1.,  1., -1., -1.,  1.],
        [ 1.,  1.,  1.,  1., -1., -1., -1., -1.]], device='npu:0')
>>> c = torch_npu.npu_sign_bits_unpack(a, 2, torch.float32)
>>> c
tensor([[ 1.,  1.,  1.,  1.,  1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1., -1.]], device='npu:0')
```

