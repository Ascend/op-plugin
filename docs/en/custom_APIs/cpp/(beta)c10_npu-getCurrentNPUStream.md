# (beta) c10_npu::getCurrentNPUStream

## Definition File

torch_npu\csrc\core\npu\NPUStream.h

## Prototype

```cpp
c10_npu::NPUStream c10_npu::getCurrentNPUStream(c10::DeviceIndex device_index = -1)
```

## Function

Obtains the current NPU stream through a device ID. The return value type is `NPUStream`, which is identical to `c10::cuda::CUDAStream c10::cuda::getCurrentCUDAStream(c10::DeviceIndex device_index = -1)`.

## Parameters

**`device_index`** (`DeviceIndex`): NPU device ID whose stream is to be obtained.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
