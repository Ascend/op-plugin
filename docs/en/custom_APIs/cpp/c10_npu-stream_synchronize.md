# c10_npu::stream_synchronize

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->

## Function

Synchronizes an NPU stream. This function is identical to `c10::cuda::stream_synchronize`.

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
void stream_synchronize(aclrtStream stream)
```

## Parameters

**`stream`** (`aclrtStream`): Required. Stream to be synchronized.

## Return Values

None

## Constraints

None
