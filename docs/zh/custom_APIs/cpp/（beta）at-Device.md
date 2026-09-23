# （beta）at::Device

## 函数原型

```cpp
at::Device(const std::string &device_string)
```

## 功能说明

在安装TorchNPU后，Device类型新增支持NPU字段，可以从字符串描述中指示设备。

## 参数说明

device_string：string类型，提供的字符串必须遵循以下架构：`(npu)[:<device-index>]`，其中NPU指定设备类型，<device-index\>可选，指定设备索引。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>
<!-- end id4 -->
