# （beta）torch_npu.contrib.module.npu_modules.DropoutWithByteMask

> [!NOTICE]
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

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

应用NPU兼容的DropoutWithByteMask操作。

## 函数原型

```python
torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5, inplace=False, max_seed=2 ** 10 - 1)
```

## 参数说明

**计算参数**

- **p** (`float`)：元素归零的概率，取值范围为[0, 1]。默认值为0.5。
- **inplace** (`bool`)：默认值为False。仅支持False，使用True进行前向计算会抛出`ValueError`。
- **max_seed** (`int`)：预留参数，暂未使用。默认值为2 ** 10 - 1（即1023）。

**计算输入**

- **Input** (`Tensor`)：输入张量，可为任何shape。

## 返回值说明

`Tensor`

输出张量与输入张量的shape相同。

## 调用示例

示例在训练模式下对一个16×16的NPU张量应用dropout。丢弃概率由构造参数`p`指定。

```python
import torch, torch_npu
from torch_npu.contrib.module.npu_modules import DropoutWithByteMask
m = DropoutWithByteMask(p=0.5)
input = torch.randn(16, 16).npu()
output = m(input)
print(output.shape)
```

结果保持16×16的形状。
