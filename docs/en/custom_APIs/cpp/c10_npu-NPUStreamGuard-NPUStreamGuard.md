# c10_npu::NPUStreamGuard::NPUStreamGuard

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Creates a stream guard. This sets the current NPU device to the device associated with the provided stream, and configures the current NPU stream. Upon destruction, the guard automatically restores the NPU device and stream to their pre-construction states. This ensures the original context is recovered when the scope exits.

## Definition File

torch_npu\csrc\core\npu\NPUGuard.h

## Prototype

```cpp
c10_npu::NPUStreamGuard::NPUStreamGuard(c10::Stream stream)
```

## Parameters

`stream` (`c10::Stream`): Required. Stream managed by the guard.

## Return Values

None

## Constraints

None
