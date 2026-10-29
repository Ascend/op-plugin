# (beta) c10_npu::setCurrentNPUStream

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

Sets the current NPU stream. This function is identical to `void c10::cuda::setCurrentCUDAStream(c10::cuda::CUDAStream stream)`.

## Definition File

torch_npu\csrc\core\npu\NPUStream.h

## Prototype

```cpp
void c10_npu::setCurrentNPUStream(c10_npu::NPUStream stream)
```

## Parameters

**`stream`** (`NPUStream`): NPU stream to be set.
