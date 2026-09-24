# （beta）torch_npu.npu.get_autocast_dtype

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

在AMP场景获取设备支持的数据类型，该`dtype`由torch_npu.npu.set_autocast_dtype设置，若未设置则使用默认数据类型`torch.float16`。

## 函数原型

```python
torch_npu.npu.get_autocast_dtype()
```

## 返回值说明

`torch.dtype`

## 调用示例

```python
import torch
import torch_npu

current_dtype = torch_npu.npu.get_autocast_dtype()

```
