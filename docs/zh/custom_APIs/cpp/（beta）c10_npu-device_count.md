# （beta）c10_npu::device_count

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

获取可使用的NPU数量，返回值类型DeviceIndex。

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```cpp
c10::DeviceIndex c10_npu::device_count()
```
