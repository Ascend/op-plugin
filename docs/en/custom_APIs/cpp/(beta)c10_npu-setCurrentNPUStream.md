# (beta) c10_npu::setCurrentNPUStream

## Definition File

torch_npu\csrc\core\npu\NPUStream.h

## Prototype

```cpp
void c10_npu::setCurrentNPUStream(c10_npu::NPUStream stream)
```

## Function

Sets the current NPU stream. This function is identical to `void c10::cuda::setCurrentCUDAStream(c10::cuda::CUDAStream stream)`.

## Parameters

**`stream`** (`NPUStream`): NPU stream to be set.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
