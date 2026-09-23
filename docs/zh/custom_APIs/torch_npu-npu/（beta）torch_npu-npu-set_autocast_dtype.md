# （beta）torch_npu.npu.set_autocast_dtype

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

设置设备在AMP场景支持的数据类型。

## 函数原型

```python
torch_npu.npu.set_autocast_dtype(dtype)
```

## 参数说明

**dtype**：数据类型。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.set_autocast_dtype(torch.float16)
```
