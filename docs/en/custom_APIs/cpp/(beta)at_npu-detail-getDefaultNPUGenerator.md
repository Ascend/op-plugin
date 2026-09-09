# (beta) at_npu::detail::getDefaultNPUGenerator

## Definition File

torch_npu\csrc\aten\NPUGeneratorImpl.h

## Prototype

```cpp
at::Generator& at_npu::detail::getDefaultNPUGenerator(c10::DeviceIndex device_index = -1)
```

## Function

Obtains the default generator for an NPU device. The return value type is `Generator`, which is identical to `at::Generator& at::cuda::detail::getDefaultCUDAGenerator(c10::DeviceIndex device_index = -1)`.

## Parameters

**`device_index`** (`DeviceIndex`): Required. NPU device ID whose generator is to be obtained.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
