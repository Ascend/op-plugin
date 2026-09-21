# （beta）torch_npu.contrib.module.SiLU

> [!NOTICE]  
>该接口计划废弃，可以使用`torch.nn.SiLU`接口进行替换。

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

按元素应用基于NPU的Sigmoid线性单元（SiLU）函数。SiLU函数也称为Swish函数。

## 函数原型

```python
torch_npu.contrib.module.SiLU(nn.Module)
```

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> from torch_npu.contrib.module import SiLU
>>> m = SiLU()
>>> input_tensor = torch.randn(2, 32, 5, 5).npu()
>>> output = m(input_tensor)
```
