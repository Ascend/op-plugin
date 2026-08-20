# c10_npu::NPUStreamGuard

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Manages the NPU device stream within its execution scope. This guard is identical to `c10::cuda::CUDAStreamGuard`.

## Definition File

torch_npu\csrc\core\npu\NPUGuard.h

## Prototype

```cpp
struct c10_npu::NPUStreamGuard
```
