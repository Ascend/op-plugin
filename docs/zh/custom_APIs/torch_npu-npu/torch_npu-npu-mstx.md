# torch_npu.npu.mstx

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

打点接口。

用于为[torch_npu.profiler._ExperimentalConfig](../torch_npu-profiler/torch_npu-profiler-_ExperimentalConfig.md)的mstx提供打点接口调用。

## 函数原型

```python
torch_npu.npu.mstx()
```

## 参数说明

无

## 返回值说明

返回`mstx`类的实例。该实例本身不保存上下文状态，用于访问 MSTX 打点相关接口。

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝运行，仅供参考。

```python
import torch
import torch_npu
mstx_object = torch_npu.npu.mstx()
```
