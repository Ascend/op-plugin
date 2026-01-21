# （beta）torch_npu.npu_silu

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch.nn.functional.silu`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

计算self的Swish。Swish是一种激活函数，计算公式为' x * sigmoid(x) '。

## 函数原型

```
torch_npu.npu_silu(self) -> Tensor
```

## 参数说明

**self**（`Tensor`）：数据类型支持`float16`、`float32`。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> a=torch.rand(2,8).npu()
>>> output = torch_npu.npu_silu(a)
>>> output
tensor([[0.4397, 0.7178, 0.5190, 0.2654, 0.2230, 0.2674, 0.6051, 0.3522],
        [0.4679, 0.1764, 0.6650, 0.3175, 0.0530, 0.4787, 0.5621, 0.4026]],
       device='npu:0')
```

