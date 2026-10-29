# c10_npu::NPUStreamGuard::reset_stream

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->

## Function

Sets a new NPU stream for the guard. This function first restores the current stream and device to their original states. It then sets the current device to the device associated with the provided stream and sets the current NPU stream on that device to the provided stream.

## Definition File

torch_npu\csrc\core\npu\NPUGuard.h

## Prototype

```cpp
void c10_npu::NPUStreamGuard::reset_stream(c10::Stream stream)
```

## Parameters

**`stream`** (`c10::Stream`): Required. Stream to be managed by the guard.

## Return Values

None

## Constraints

`stream` must be an NPU stream (a `c10::Stream` created by an NPU device). Otherwise, the behavior is undefined.
