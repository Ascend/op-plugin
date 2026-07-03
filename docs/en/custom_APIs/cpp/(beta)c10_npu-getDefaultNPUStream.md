# (beta) c10_npu::getDefaultNPUStream

## Definition File

torch_npu\csrc\core\npu\NPUStream.h

## Prototype

```cpp
c10_npu::NPUStream c10_npu::getDefaultNPUStream(c10::DeviceIndex device_index = -1)
```

## Function

Obtains the default NPU stream. The return value type is `NPUStream`, which is identical to `c10::cuda::CUDAStream c10::cuda::getDefaultCUDAStream(c10::DeviceIndex device_index = -1)`.

## Parameters

**`device_index`** (`DeviceIndex`): NPU device ID whose stream is to be obtained. The default value is `-1`, which specifies to use the current NPU device.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
