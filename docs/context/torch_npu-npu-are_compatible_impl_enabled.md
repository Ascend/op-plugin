# torch_npu.npu.are_compatible_impl_enabled

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term>                        |    √     |
| <term>Atlas A3 推理系列产品</term>                        |    √     |
| <term>Atlas A2 训练系列产品</term>                        |    √     |
| <term>Atlas A2 推理系列产品</term>                        |    √     |
| <term>Atlas 推理系列产品</term>                           |    √     |
| <term>Atlas 训练系列产品</term>                           |    √     |

## 功能说明

该接口用于查询`torch_npu.npu.use_compatible_impl`的配置情况，查看算子API的实现是否与社区完全对齐。

## 函数原型

```
torch_npu.npu.are_compatible_impl_enabled()
```

## 参数说明

无

## 返回值说明
`bool`

True为已开启，False为未开启。



## 约束说明

无

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.use_compatible_impl(True)
>>> torch_npu.npu.are_compatible_impl_enabled()
True
```