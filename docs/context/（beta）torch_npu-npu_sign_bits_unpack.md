# （beta）torch\_npu.npu\_sign\_bits\_unpack

## 函数原型

```
torch_npu.npu_sign_bits_unpack(x, size, dtype) -> Tensor
```

## 功能说明

将uint8类型1位Adam拆包为float。

## 参数说明

-   x\(Tensor\) - 1D uint8张量。
-   size\(Int\) - reshape时输出张量的第一个维度。
-   dtype\(torch.dtype\) - 值为torch.float16设置输出类型为float16，值为torch.float32设置输出类型为float32。

## 约束说明

size可被uint8拆包的输出整除。输出大小为\(size of x\) \* 8。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> a = torch.tensor([159, 15], dtype=torch.uint8).npu()
>>> b = torch_npu.npu_sign_bits_unpack(a, 2, torch.float32)
>>> b
tensor([[ 1.,  1.,  1.,  1.,  1., -1., -1.,  1.],
        [ 1.,  1.,  1.,  1., -1., -1., -1., -1.]], device='npu:0')
```

