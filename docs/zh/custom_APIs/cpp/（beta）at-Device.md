# （beta）at::Device

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

在安装TorchNPU后，Device类型新增支持NPU字段，可以从字符串描述中指定设备。

## 定义文件

`c10/core/Device.h`（PyTorch头文件）

## 函数原型

```cpp
at::Device(const std::string &device_string)
```

## 参数说明

**device_string** (`const std::string&`)：必选参数。设备字符串格式为`npu`或`npu:<device-index>`。例如，`at::Device("npu:0")`指定第0个可见NPU，`at::Device("npu")`仅指定设备类型。使用显式索引访问设备时，索引须满足`0 <= device-index < c10_npu::device_count()`。
