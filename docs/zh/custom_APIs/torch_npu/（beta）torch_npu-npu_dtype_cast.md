# （beta）torch_npu.npu_dtype_cast

> [!NOTICE]  
> 该接口计划废弃，可以使用`tensor.to()`接口进行替换。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

执行张量数据类型（dtype）转换。支持FakeTensor模式。

## 函数原型

```python
torch_npu.npu_dtype_cast(input, dtype) -> Tensor
```

## 参数说明

- **input**（`Tensor`）：输入张量。
- **dtype**（`torch.dtype`）：返回张量的目标数据类型。

## 调用示例

示例一：

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu_dtype_cast(torch.tensor([0, 0.5, -1.]).npu(), dtype=torch.int)
tensor([ 0,  0, -1], device='npu:0', dtype=torch.int32)
```

示例二：

```python
#FakeTensor模式
>>> import torch
>>> import torch_npu
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     x = torch.rand(2, dtype=torch.float32).npu()
...     res = torch_npu.npu_dtype_cast(x, torch.float16)
...
>>> print(res)
FakeTensor(..., device='npu:0', size=(2,), dtype=torch.float16)
```
