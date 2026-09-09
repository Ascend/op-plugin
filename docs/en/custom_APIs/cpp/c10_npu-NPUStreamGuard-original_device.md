# c10_npu::NPUStreamGuard::original_device

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Returns the NPU device captured during guard construction.

## Definition File

torch_npu\csrc\core\npu\NPUGuard.h

## Prototype

```cpp
c10::Device c10_npu::NPUStreamGuard::original_device() const
```

## Parameters

None

## Return Values

`c10::Device`

NPU device captured during guard construction.

## Constraints

None
