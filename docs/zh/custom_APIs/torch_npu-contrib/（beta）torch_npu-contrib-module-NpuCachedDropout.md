# （beta）torch_npu.contrib.module.NpuCachedDropout

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

在NPU设备上使用FairseqDropout。

## 函数原型

```python
torch_npu.contrib.module.NpuCachedDropout(p, module_name=None)
```

## 参数说明

- **p** (`float`)：元素归零的概率，取值范围为 [0, 1]，无默认值（必填）。例如p=0.5表示每个元素以50%的概率被置零。
- **module_name** (`str`)：可选参数，模块名称，用于标识当前Dropout所属的模块，默认为`None`。该参数仅作标识用途，不参与实际计算逻辑。

## 约束说明

不支持动态shape。

## 成员函数

`NpuCachedDropout.enable_dropout_ensemble(model)`接收包含`NpuCachedDropout`的`torch.nn.Module`，为模型注册生成和缓存随机掩码的钩子。请在前向计算前调用。

## 调用示例

示例将一个丢弃概率为0.5的模块接入`Sequential`，并在前向前注册掩码钩子。第一次前向用于准备与输入形状匹配的掩码，第二次前向使用缓存掩码。

```python
import torch
import torch.nn as nn
import torch_npu
from torch_npu.contrib.module import NpuCachedDropout
dropout = NpuCachedDropout(0.5)
model = nn.Sequential(dropout).npu()
NpuCachedDropout.enable_dropout_ensemble(model)
x = torch.randn(2, 3, 4, 4).npu()
_ = model(x)
output = model(x)
print(output.shape)
```

示例输出如下。

```text
torch.Size([2, 3, 4, 4])
```
