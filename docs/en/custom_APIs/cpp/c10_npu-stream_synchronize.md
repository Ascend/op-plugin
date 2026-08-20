# c10_npu::stream_synchronize

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>            |    √     |
|<term>Atlas A2 training products</term>  | √   |

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
