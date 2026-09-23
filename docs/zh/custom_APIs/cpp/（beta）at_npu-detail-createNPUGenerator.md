# （beta）at_npu::detail::createNPUGenerator

## 定义文件

torch_npu\csrc\aten\NPUGeneratorImpl.h

## 函数原型

```cpp
at::Generator at_npu::detail::createNPUGenerator(c10::DeviceIndex device_index = -1)
```

## 功能说明

用于创建NPU设备默认生成器，返回值类型Generator，与at::Generator at::cuda::detail::createCUDAGenerator(c10::DeviceIndex _device_index_ = -1)相同。

## 参数说明

device_index：DeviceIndex类型，指定创建生成器的NPU设备ID。

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
