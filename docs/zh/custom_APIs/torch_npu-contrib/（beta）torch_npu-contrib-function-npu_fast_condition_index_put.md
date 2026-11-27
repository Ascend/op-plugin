# （beta）torch_npu.contrib.function.npu_fast_condition_index_put

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

使用NPU亲和写法替换bool型index_put函数中的原生写法。

## 函数原型

```python
torch_npu.contrib.function.npu_fast_condition_index_put(x, condition, value)
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入张量。
- **condition** (`torch.bool`类型的Tensor)：必选参数，替换条件，须与`x`在同一设备且形状可广播。
- **value** (`int | float`)：必选参数，条件为`True`的位置使用的标量替换值。其类型按PyTorch标量提升规则与输入张量共同参与计算。

## 返回值说明

`Tensor`

返回经过条件替换后的张量。

## 调用示例

示例把输入中小于0.5的元素替换为0，并与同形状布尔掩码的索引赋值写法比较。

```python
import torch
from torch_npu.contrib.function import npu_fast_condition_index_put
import copy
x = torch.randn(128, 8192).npu()
condition = x < 0.5
value = 0.
x1 = copy.deepcopy(x)
x1[condition] = value
x1_opt = npu_fast_condition_index_put(x, condition, value)
print(torch.equal(x1, x1_opt))
print(x1_opt.shape)
```

示例输出如下。

```text
True
torch.Size([128, 8192])
```
