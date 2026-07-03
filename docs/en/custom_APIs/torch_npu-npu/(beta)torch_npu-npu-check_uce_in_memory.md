# (beta) torch_npu.npu.check_uce_in_memory

> [!NOTICE]  
> This API is reserved and is not supported currently.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Provides a faulty memory address type detection interface for MindCluster to determine fault recovery strategies. It is used to determine the type of a faulty memory address when a UCE on-chip memory fault occurs.

> [!CAUTION]  
> The implementation of this API depends on the PyTorch memory management mechanism. This API is available only when the memory reuse mechanism is enabled (when `PYTORCH_NO_NPU_MEMORY_CACHING` is not configured). If `export PYTORCH_NO_NPU_MEMORY_CACHING=1` is set, the memory reuse mechanism is disabled and this API becomes unavailable.

## Prototype

```python
torch_npu.npu.check_uce_in_memory(device_id:int)
```

## Parameters

**`device_id`** (`int`): ID of the device to be processed.

## Return Values

- `0`: No UCE fault address exists.
- `1`: The UCE fault address is not an Ascend Extension for PyTorch memory address.
- `2`: The UCE fault address is a temporary memory address used by Ascend Extension for PyTorch.
- `3`: The UCE fault address is a resident memory address used by Ascend Extension for PyTorch.

## Constraints

The specified device ID must be valid.

## Example

 ```python
>>> import torch,torch_npu
>>> torch.npu.set_device(0)
>>> torch_npu.npu.check_uce_in_memory(0)
 ```
