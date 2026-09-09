# (beta) c10_npu::getNPUStreamFromPool

## Definition File

torch_npu\csrc\core\npu\NPUStream.h

## Prototype

```cpp
c10_npu::NPUStream c10_npu::getNPUStreamFromPool(c10::DeviceIndex device = -1)
```

## Function

Obtains a new stream from the NPU stream pool. Streams are preallocated in the pool and retrieved in a round-robin manner. The return value type is `NPUStream`.

## Parameters

**`device`** (`DeviceIndex`): NPU device ID whose stream is to be obtained.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
