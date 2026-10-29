# c10_npu::NPUStreamGuard

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->

## Function

Manages the NPU device stream within its execution scope. This guard is identical to `c10::cuda::CUDAStreamGuard`.

## Definition File

torch_npu\csrc\core\npu\NPUGuard.h

## Prototype

```cpp
struct c10_npu::NPUStreamGuard
```
