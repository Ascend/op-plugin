# （beta）torch_npu.contrib.function.npu_fast_condition_index_put

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

使用NPU亲和写法替换bool型index_put函数中的原生写法。

## 函数原型

```
torch_npu.contrib.function.npu_fast_condition_index_put(x, condition, value)
```

## 参数说明

- **x** (`Tensor`)：输入张量。
- **condition** (`BoolTensor`)：判断条件。
- **value** (`int`)：bboxes步长。

## 返回值说明

`Tensor`

代表框转换deltas。

## 调用示例

```python
>>> import torch
>>> from torch_npu.contrib.function import npu_fast_condition_index_put
>>> import copy
>>> x = torch.randn(128, 8192).npu()
>>> condition = x < 0.5
>>> value = 0.
>>> x1 = copy.deepcopy(x)[condition] = value
>>> x1_opt = npu_fast_condition_index_put(x, condition, value)
>>> x1_opt
tensor([[0.9661, 1.6750, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [1.3621, 0.0000, 0.9606,  ..., 1.4324, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0000, 0.0000,  ..., 1.4605, 0.7734, 1.9721],
        [0.0000, 0.9325, 0.7112,  ..., 0.0000, 0.9814, 1.4227],
        [1.0037, 0.0000, 0.0000,  ..., 0.0000, 1.6497, 0.0000]],
       device='npu:0')
>>> x1_opt.shape
torch.Size([128, 8192])

```

