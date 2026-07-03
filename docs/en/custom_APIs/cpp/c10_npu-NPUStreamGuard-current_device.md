# c10_npu::NPUStreamGuard::current_device

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Returns the latest NPU device set by this stream guard.

## Definition File

torch_npu\csrc\core\npu\NPUGuard.h

## Prototype

```cpp
c10::Device c10_npu::NPUStreamGuard::current_device() const
```

## Parameters

None

## Return Values

`c10::Device`

The current device.

## Constraints

None
