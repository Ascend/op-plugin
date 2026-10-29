# c10_npu::NPUStreamGuard::original_stream

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->

## Function

Returns the NPU stream set during guard construction.

## Definition File

torch_npu\csrc\core\npu\NPUGuard.h

## Prototype

```cpp
c10_npu::NPUStream c10_npu::NPUStreamGuard::original_stream() const
```

## Parameters

None

## Return Values

`c10_npu::NPUStream`

Stream set during guard construction.

## Constraints

None
