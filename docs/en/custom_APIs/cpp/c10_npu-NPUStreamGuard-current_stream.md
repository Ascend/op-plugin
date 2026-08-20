# c10_npu::NPUStreamGuard::current_stream

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Returns the latest NPU stream set by this stream guard.

## Definition File

torch_npu\csrc\core\npu\NPUGuard.h

## Prototype

```cpp
c10_npu::NPUStream c10_npu::NPUStreamGuard::current_stream() const
```

## Parameters

None

## Return Values

`c10_npu::NPUStream`

The currently managed stream.

## Constraints

None
