# torch_npu.optim.NpuFusedAdamP

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

通过张量融合实现的高性能AdamP优化器。AdamP的功能和原理可参考《[AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights](https://arxiv.org/pdf/2006.08217)》。

## 函数原型

```python
class torch_npu.optim.NpuFusedAdamP(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False)
```

## 参数说明

- **params** (`iterable`)：必选参数，模型参数或模型参数组。
- **lr** (`float`)：可选参数，学习率，默认值为1e-3。
- **betas** (`Tuple[float, float]`)：可选参数，用于计算梯度及其平方的运行平均值的系数，`betas`为包含两个值的tuple类型，默认值为(0.9, 0.999)。
- **eps** (`float`)：可选参数，作为分母防止除零错误，提高数值稳定性，默认值为1e-8。
- **weight_decay** (`float`)：可选参数，权重衰减，默认值为0。
- **delta** (`float`)：可选参数，余弦相似度阈值，默认值为0.1。
- **wd_ratio** (`float`)：权重衰减动态调整速率，默认值为0.1。
- **nesterov** (`bool`)：使用Nesterov动量，默认值为False。

## 返回值说明

类型为`NpuFusedAdamP`的对象。

## 约束说明

`NpuFusedAdamP`的实现机制要求`params`中的每一个模型参数对象在使用过程中不能被重新申请，否则将导致无法预料的结果。引起模型参数对象被重新申请的操作包括但不限于：

- 将模型或其子Module执行`.cpu()`操作。
- 将模型参数对象指向新的对象。
- 将模型参数对象置为`None`。

对模型参数对象进行inplace计算，或者读取参数的值，`NpuFusedAdamP`可正常工作。

## 调用示例

构造一个NPU参数及其模拟梯度，执行一次优化器更新。

```python
import torch
from torch_npu.optim import NpuFusedAdamP

param = torch.nn.Parameter(torch.tensor([1.0, 2.0], device="npu"))
param.grad = torch.tensor([0.01, -0.02], device="npu")
before = param.detach().clone()

optimizer = NpuFusedAdamP([param], eps=1e-5, betas=(0.9, 0.999), lr=2e-3, weight_decay=0.05)
optimizer.step()
assert not torch.equal(before, param.detach())
```
