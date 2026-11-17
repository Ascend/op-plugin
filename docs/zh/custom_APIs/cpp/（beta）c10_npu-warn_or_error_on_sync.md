# （beta）c10_npu::warn_or_error_on_sync

## 产品支持情况

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

当NPU同步时，该接口根据当前警告等级决定报错还是仅告警，并且无返回值，其行为与void c10::cuda::warn_or_error_on_sync()相同。

## 定义文件

torch_npu/csrc/core/npu/NPUFunctions.h

## 函数原型

```cpp
void c10_npu::warn_or_error_on_sync()
```
