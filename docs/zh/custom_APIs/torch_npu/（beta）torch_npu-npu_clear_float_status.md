# （beta）torch_npu.npu_clear_float_status

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

清除溢出检测相关标志位。

## 函数原型

```python
torch_npu.npu_clear_float_status(self) -> Tensor
```

## 参数说明

**self**(`Tensor`)：数据类型为`float32`的张量。

## 返回值说明

`Tensor`

一个包含8个float32类型全零值的Tensor。

## 调用示例

```python
>>> import torch, torch_npu
>>> x = torch.rand(2).npu()
>>> torch_npu.npu_clear_float_status(x)
tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
```
