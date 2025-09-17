# （beta）torch_npu.npu_sign_bits_pack

## 函数原型

```
torch_npu.npu_sign_bits_pack(Tensor self, int size) -> Tensor
```

## 功能说明

将float类型1位Adam打包为uint8。

-   API功能：将float类型的输入打包为uint8类型。每8个浮点数打包为一个uint8数值，-1.0编码为二进制位0，1.0编码为二进制位1，并按小端序进行打包。

-   小算子等价计算逻辑：
    
     可使用`sign_pack`等价替换`torch_npu.npu_sign_bits_pack`，两者计算逻辑一致。
     
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
## 参数说明

- self(Tensor): 必选参数，1D float张量。 支持float32和float16类型输入。
- size(Int): 必选参数，用于reshape输出张量的第一个维度。

## 返回值说明
`Tensor`

打包后的张量。

## 约束说明

size可被float打包的输出整除。如果self的size可被8整除，则输出的size为(size of self)/8；否则，输出的size为(size of self // 8) + 1。将在小端位置添加-1浮点值以填充可整除性。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> import torch, torch_npu
>>> a = torch.tensor([5,4,3,2,0,-1,-2, 4,3,2,1,0,-1,-2],dtype=torch.float32).npu()
>>> b = torch_npu.npu_sign_bits_pack(a, 2)
>>> b
tensor([[159],
        [ 15]], device='npu:0', dtype=torch.uint8)
>>> c = torch_npu.npu_sign_bits_pack(a, 1)
>>> c
tensor([[159, 15]], device='npu:0', dtype=torch.uint8)
```

