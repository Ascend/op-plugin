# （beta）torch_npu.contrib.function.npu_fast_condition_index_put

## 函数原型

```
torch_npu.contrib.function.npu_fast_condition_index_put(x, condition, value)
```

## 功能说明

使用NPU亲和写法替换bool型index_put函数中的原生写法。

## 参数说明

- x (torch.Tensor) - normal Tensor。
- condition (torch.BoolTensor) - 判断条件。
- value (int, float) - bboxes步长。

## 输出说明

torch.Tensor - 框转换deltas。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

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

