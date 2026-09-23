# （beta）torch_npu.npu.get_mm_bmm_format_nd

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

确认线性module里面的mm和bmm算子是否已启用ND格式，如果启用了ND，返回True，否则返回False。

## 函数原型

```python
torch_npu.npu.get_mm_bmm_format_nd()
```

## 返回值说明

`bool`

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.get_mm_bmm_format_nd()
True
```
