# （beta）torch_npu::init_npu

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

初始化NPU设备。

## 定义文件

torch_npu/csrc/libs/init_npu.h

## 函数原型

```cpp
void torch_npu::init_npu(const c10::DeviceIndex device_index = 0)
void torch_npu::init_npu(const std::string& device_str)
void torch_npu::init_npu(const at::Device& device)
```

## 参数说明

- **device_index** (`c10::DeviceIndex`)：索引重载的可选参数，默认值为`0`。显式指定的NPU设备ID须满足`0 <= device_index < c10_npu::device_count()`。
- **device_str** (`const std::string&`)：字符串重载的必选参数，例如`torch_npu::init_npu("npu:0")`。
- **device** (`const at::Device&`)：设备对象重载的必选参数，设备类型须为NPU，例如`torch_npu::init_npu(at::Device("npu:0"))`。
