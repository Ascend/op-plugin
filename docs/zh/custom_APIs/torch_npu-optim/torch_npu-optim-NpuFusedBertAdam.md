# torch_npu.optim.NpuFusedBertAdam

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

通过张量融合实现的高性能BertAdam优化器。BertAdam的功能和原理可参考[https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/optimization.py\#L64](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/optimization.py#L64)。

## 函数原型

```python
class torch_npu.optim.NpuFusedBertAdam(params, lr=required, warmup=-1, t_total=-1, schedule="warmup_linear", b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0)
```

## 参数说明

- **params** (`iterable`)：必选参数，模型参数或模型参数组。
- **lr** (`float`)：可选参数，指定各参数组的默认学习率。省略此参数时，每个参数组都必须指定`lr`。传入负值时抛出`ValueError`。
- **warmup** (`float`)：可选参数，`t_total`的warmup比例，默认值为-1，表示不进行warmup。`warmup`的值小于0且`warmup`不等于-1，或者`warmup`大于等于1，抛出“ValueError”异常信息。
- **t_total** (`float`)：可选参数，学习率调整的总步数。默认值为-1，表示固定学习率。
- **schedule** (`str`)：可选参数，学习率warmup策略，默认值为warmup_linear。`schedule`为字符串，其值必须为warmup_cosine、warmup_constant、warmup_linear、warmup_poly中的一个。
- **b1** (`float`)：可选参数，Adam b1，默认值为0.9。`b1`的值小于0或大于等于1时，抛出“ValueError”异常信息。
- **b2** (`float`)：可选参数，Adam b2，默认值为0.999。`b2`的值小于0或大于等于1时，抛出“ValueError”异常信息。
- **e** (`float`)：可选参数，Adam epsilon，默认值为1e-6。`e`的值小于0时，抛出“ValueError”异常信息。
- **weight_decay** (`float`)：可选参数，权重衰减，默认值为0.01。
- **max_grad_norm** (`float`)：可选参数，最大梯度范数，默认值为1.0，小于或等于0时不做裁剪。

## 返回值说明

类型为`NpuFusedBertAdam`的对象。

## 约束说明

`NpuFusedBertAdam`的实现机制要求`params`中的每一个模型参数对象在使用过程中不能被重新申请，否则将导致无法预料的结果。引起模型参数对象被重新申请的操作包括但不限于：

- 将模型或其子Module执行`.cpu()`操作。
- 将模型参数对象指向新的对象。
- 将模型参数对象置为`None`。

对模型参数对象进行inplace计算，或者读取参数的值，`NpuFusedBertAdam`可正常工作。

## 调用示例

构造一个NPU参数及其模拟梯度，执行一次优化器更新。

```python
import torch
from torch_npu.optim import NpuFusedBertAdam

param = torch.nn.Parameter(torch.tensor([1.0, 2.0], device="npu"))
param.grad = torch.tensor([0.01, -0.02], device="npu")

optimizer = NpuFusedBertAdam([param], lr=0.01, warmup=0.1, t_total=20, max_grad_norm=-1)
optimizer.step()
# 启用 warmup 后，首次 step 的调度学习率可能为 0。
```
