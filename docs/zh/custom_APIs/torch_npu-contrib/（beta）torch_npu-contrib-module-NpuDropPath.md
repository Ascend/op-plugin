# （beta）torch_npu.contrib.module.NpuDropPath

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

用于在NPU上实现随机深度。在训练前通过`NpuDropPath.enable_droppath_ensemble(model)`注册随机张量生成钩子。模块预热时先记录输入形状，随机张量队列就绪后按样本丢弃主路径输出。

## 函数原型

```python
torch_npu.contrib.module.NpuDropPath(drop_prob=None)
```

## 参数说明

**计算参数**

- **drop_prob** (`float`)：调用时须显式给出，表示路径丢弃概率，取值范围为 [0, 1)。设为0时直接返回输入。

**计算输入**

- **x** (`Tensor`)：应用DropPath的输入张量。

## 返回值说明

`Tensor`

DropPath的计算结果。

## 调用示例

示例将`input1`用作残差分支，将`input2`用作主路径。将丢弃概率设为`0`后，两条分支的输出直接相加。

```python
import torch, torch_npu
from torch_npu.contrib.module import NpuDropPath
input1 = torch.randn(68, 5).npu()
input1.requires_grad_(True)
input2 = torch.randn(68, 5).npu()
input2.requires_grad_(True)
fast_drop_path = NpuDropPath(0).npu()
output = input1 + fast_drop_path(input2)
output.sum().backward()
```

输出保持 `(68, 5)`形状，`backward()`会计算两条分支的梯度。
