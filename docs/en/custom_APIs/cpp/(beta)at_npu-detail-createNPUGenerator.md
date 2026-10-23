# (beta) at_npu::detail::createNPUGenerator

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->

## Function

Creates the default generator for an NPU device. The return value type is `Generator`, which is identical to `at::Generator at::cuda::detail::createCUDAGenerator(c10::DeviceIndex _device_index_ = -1)`.

## Definition File

torch_npu\csrc\aten\NPUGeneratorImpl.h

## Prototype

```cpp
at::Generator at_npu::detail::createNPUGenerator(c10::DeviceIndex device_index = -1)
```

## Parameters

`device_index` (`DeviceIndex`): NPU device ID for which the generator is created.
