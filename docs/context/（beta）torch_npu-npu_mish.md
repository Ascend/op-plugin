# （beta）torch_npu.npu_mish

>**须知：**<br>
>该接口计划废弃，可以使用`torch.nn.functional.mish`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

按元素计算self的双曲正切。

## 函数原型

```
torch_npu.npu_mish(self) -> Tensor
```

## 参数说明

**self**（`Tensor`）：数据类型支持`float16`、`float32`。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> x = torch.rand(10, 30, 10).npu()
>>> y = torch_npu.npu_mish(x)
>>> y.shape
torch.Size([10, 30, 10])
```

