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

- **x** (`Tensor`)：normal Tensor。
- **condition** (`BoolTensor`)：判断条件。
- **value** (`int`)：bboxes步长。

## 返回值说明

`Tensor`

代表框转换deltas。

## 调用示例

```python
>>> from torch_npu.contrib.function import npu_fast_condition_index_put
>>> import copy
>>> x = torch.randn(128, 8192).npu()
>>> condition = x < 0.5
>>> value = 0.
>>> x1 = copy.deepcopy(x)[condition] = value
>>> x1_opt = npu_fast_condition_index_put(x, condition, value)
```

