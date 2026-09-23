# torch_npu.npu.are_compatible_impl_enabled

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id7 -->

## 功能说明

该接口用于查询`torch_npu.npu.use_compatible_impl`的配置情况，查看算子API的实现是否与社区完全对齐。

## 函数原型

```python
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
