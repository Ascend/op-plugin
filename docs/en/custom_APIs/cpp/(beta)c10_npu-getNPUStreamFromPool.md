# (beta) c10_npu::getNPUStreamFromPool

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

Obtains a new stream from the NPU stream pool. Streams are preallocated in the pool and retrieved in a round-robin manner. The return value type is `NPUStream`.

## Definition File

torch_npu\csrc\core\npu\NPUStream.h

## Prototype

```cpp
c10_npu::NPUStream c10_npu::getNPUStreamFromPool(c10::DeviceIndex device = -1)
```

## Parameters

**`device`** (`DeviceIndex`): NPU device ID whose stream is to be obtained.
