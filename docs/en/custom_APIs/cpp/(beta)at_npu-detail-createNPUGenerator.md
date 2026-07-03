# (beta) at_npu::detail::createNPUGenerator

## Definition File

torch_npu\csrc\aten\NPUGeneratorImpl.h

## Prototype

```cpp
at::Generator at_npu::detail::createNPUGenerator(c10::DeviceIndex device_index = -1)
```

## Function

Creates the default generator for an NPU device. The return value type is `Generator`, which is identical to `at::Generator at::cuda::detail::createCUDAGenerator(c10::DeviceIndex device_index = -1)`.

## Parameters

`device_index` (`DeviceIndex`): NPU device ID for which the generator is created.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
