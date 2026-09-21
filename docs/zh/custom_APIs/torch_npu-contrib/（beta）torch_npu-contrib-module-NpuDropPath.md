# （beta）torch_npu.contrib.module.NpuDropPath

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

使用NPU亲和写法替换swin_transformer.py中的原生DropPath。丢弃每个样本（应用于residual blocks的主路径）的路径（随机深度）。

## 函数原型

```python
torch_npu.contrib.module.NpuDropPath(drop_prob=None)
```

## 参数说明

**计算参数**

- **drop_prob** (`float`)：DropPath概率（路径丢弃概率）。

**计算输入**

- **x** (`Tensor`)：应用DropPath的输入张量。

## 返回值说明

`Tensor`

DropPath的计算结果。

## 调用示例

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module import NpuDropPath
>>> input1 = torch.randn(68, 5).npu()
>>> input1.requires_grad_(True)
>>> input2 = torch.randn(68, 5).npu()
>>> input2.requires_grad_(True)
>>> fast_drop_path = NpuDropPath(0).npu()
>>> output = input1 + fast_drop_path(input2)
>>> output.sum().backward()
```
