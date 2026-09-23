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

- **x** (`Tensor`)：输入张量。
- **condition** (`BoolTensor`)：判断条件。
- **value** (`int`/`float`)：满足条件时用于替换的值。

## 返回值说明

`Tensor`

返回经过条件替换后的张量。

## 调用示例

```python
>>> import torch
>>> from torch_npu.contrib.function import npu_fast_condition_index_put
>>> import copy
>>> x = torch.randn(128, 8192).npu()
>>> condition = x < 0.5
>>> value = 0.
>>> x1 = copy.deepcopy(x)
>>> x1[condition] = value
>>> x1_opt = npu_fast_condition_index_put(x, condition, value)
>>> print(x1_opt)
tensor([[0.9661, 1.6750, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [1.3621, 0.0000, 0.9606,  ..., 1.4324, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0000, 0.0000,  ..., 1.4605, 0.7734, 1.9721],
        [0.0000, 0.9325, 0.7112,  ..., 0.0000, 0.9814, 1.4227],
        [1.0037, 0.0000, 0.0000,  ..., 0.0000, 1.6497, 0.0000]],
       device='npu:0')
>>> print(x1_opt.shape)
torch.Size([128, 8192])

```
