# (beta) c10_npu::getDefaultNPUStream

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

Obtains the default NPU stream. The return value type is `NPUStream`. Its function and usage are identical to `c10::cuda::CUDAStream c10::cuda::getDefaultCUDAStream(c10::DeviceIndex device_index = -1)`.

## Definition File

torch_npu\csrc\core\npu\NPUStream.h

## Prototype

```cpp
c10_npu::NPUStream c10_npu::getDefaultNPUStream(c10::DeviceIndex device_index = -1)
```

## Parameters

**`device_index`** (`DeviceIndex`): NPU device ID whose stream is to be obtained. The default value is `-1`, which specifies to use the current NPU device.
