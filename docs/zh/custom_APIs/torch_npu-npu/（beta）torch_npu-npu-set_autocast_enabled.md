# （beta）torch_npu.npu.set_autocast_enabled

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

在设备上开启或关闭AMP。

## 函数原型

```python
torch_npu.npu.set_autocast_enabled(bool)
```

## 参数说明

**bool**：入参为True时，在设备上开启AMP，否则，不开启AMP。

## 调用示例

```python
import torch
import torch_npu
torch_npu.npu.set_autocast_enabled(True)
```
