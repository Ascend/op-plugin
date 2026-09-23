# （beta）at_npu::detail::getDefaultNPUGenerator

## 定义文件

torch_npu\csrc\aten\NPUGeneratorImpl.h

## 函数原型

```cpp
at::Generator& at_npu::detail::getDefaultNPUGenerator(c10::DeviceIndex device_index = -1)
```

## 功能说明

NPU设备默认生成器获取，返回值类型Generator，与at::Generator& at::cuda::detail::getDefaultCUDAGenerator(c10::DeviceIndex device_index = -1)相同。

## 参数说明

device_index：DeviceIndex类型，指定获取生成器的NPU设备id。

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
