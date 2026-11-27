# （beta）torch\_npu.npu\_sign\_bits\_unpack

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

- API功能：将`torch.uint8`输入按低位到高位的顺序拆包，每个字节解码为8个浮点数，0解码为-1.0，1解码为1.0。输出类型由`dtype`指定，支持`torch.float16`和`torch.float32`。

- 等价计算逻辑：
    
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

## 函数原型

```python
torch_npu.npu_sign_bits_unpack(x, size, dtype) -> Tensor
```

## 参数说明

- **x** (`Tensor`)：必选参数，1D `torch.uint8`张量。
- **size** (`int`)：必选参数，用于reshape输出张量的第一个维度。
- **dtype** (`torch.dtype`)：必选参数，值为`torch.float16`时设置输出类型为`torch.float16`，值为`torch.float32`时设置输出类型为`torch.float32`。

## 返回值说明

`Tensor`

拆包后的张量。

## 约束说明

`x`的`torch.uint8`拆包输出大小应能被`size`整除。`x`的`torch.uint8`拆包输出大小为(size of x) * 8。

## 调用示例

示例将uint8张量中的符号位展开，再按size重排输出并转换为所选浮点类型。

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
