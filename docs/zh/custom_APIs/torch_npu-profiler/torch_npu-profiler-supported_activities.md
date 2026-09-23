# torch_npu.profiler.supported_activities

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
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

查询当前支持采集的activities参数的CPU、NPU事件。

## 函数原型

```python
torch_npu.profiler.supported_activities()
```

## 返回值说明

返回一个`set`集合，包含当前支持采集的activities类型。成功时返回`{<ProfilerActivity.CPU: 1>, <ProfilerActivity.NPU: 2>}`，表示当前支持采集CPU事件和NPU事件；失败时返回空集合`set()`。

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝运行，仅供参考。

```python
import torch
import torch_npu

...

torch_npu.profiler.supported_activities()
```
