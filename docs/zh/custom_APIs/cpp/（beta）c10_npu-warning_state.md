# （beta）c10_npu::warning_state

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

返回同步调试状态对象的引用，类型为`c10_npu::WarningState&`。调用该对象的`get_sync_debug_mode()`可获取当前警告等级，类型为`SyncDebugMode`，取值为`L_DISABLED`、`L_WARN`或`L_ERROR`。

## 定义文件

torch_npu/csrc/core/npu/NPUFunctions.h

## 函数原型

```cpp
c10_npu::WarningState& c10_npu::warning_state()
```
