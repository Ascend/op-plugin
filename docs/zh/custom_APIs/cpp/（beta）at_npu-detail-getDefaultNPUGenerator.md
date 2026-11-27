# （beta）at_npu::detail::getDefaultNPUGenerator

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

获取指定NPU设备的默认随机数生成器，返回`const at::Generator&`。

## 定义文件

torch_npu/csrc/aten/NPUGeneratorImpl.h

## 函数原型

```cpp
const at::Generator& at_npu::detail::getDefaultNPUGenerator(c10::DeviceIndex device_index = -1)
```

## 参数说明

**device_index** (`c10::DeviceIndex`)：可选参数，指定获取生成器的NPU设备ID。默认值 `-1`表示当前NPU设备。若显式指定设备ID，则须满足`0 <= device_index < c10_npu::device_count()`。
