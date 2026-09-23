# （beta）torch_npu::init_npu

## 定义文件

torch_npu\csrc\libs\init_npu.h

## 函数原型

```cpp
void torch_npu::init_npu(const c10::DeviceIndex device_index = 0)
void torch_npu::init_npu(const std::string& device_str)
void torch_npu::init_npu(const at::Device& device)
```

## 功能说明

初始化NPU设备。

## 参数说明

- device_index：DeviceIndex类型，指定初始化的NPU设备ID，默认0。
- device_str：string类型，指定初始化的设备名称。
- device：Device类型，指定初始化的NPU设备。

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
