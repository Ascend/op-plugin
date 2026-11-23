# （beta）torch_npu.npu_sign_bits_pack

> [!NOTICE]  
> 该接口计划废弃，底层算子Kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

- API功能：将一维`torch.float16`或`torch.float32`张量按符号位打包为`torch.uint8`。每8个浮点数打包成1个字节，负数编码为0，零和正数编码为1，按低位到高位的顺序写入。

- 等价计算逻辑：
    
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

## 函数原型

```python
torch_npu.npu_sign_bits_pack(self, size) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，一维张量，支持`torch.float16`和`torch.float32`。
- **size** (`int`)：必选参数，为正整数，指定输出张量的第一维长度，需满足约束说明中的整除要求。

## 返回值说明

`Tensor`

打包后的张量。

## 约束说明

`size`必须能整除打包后的输出长度（即打包后的输出长度可被`size`整除）。设打包后的字节数为`L = (self.numel() + 7) // 8`。`size`必须整除`L`。当最后一组不足8个元素时，将最后一个字节的剩余高位补0。

## 调用示例

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
