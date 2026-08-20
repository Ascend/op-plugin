# torch_npu.npu.empty_virt_addr_cache

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Provides a lightweight cache release API corresponding to `torch.npu.empty_cache`. This API releases only virtual memory and unmaps it from physical memory without actually releasing the physical memory, thereby reducing the call duration.

## Definition File

torch_npu/npu/memory.py

## Prototype

```python
torch_npu.npu.empty_virt_addr_cache() -> None
```

## Parameters

None

## Return Values

None

## Constraints

This API takes effect only when the environment variable `PYTORCH_NPU_ALLOC_CONF` is set to `expandable_segments:True`. Otherwise, a runtime error will be raised.

## Example

```python
>>> import torch
>>> import torch_npu
>>> x = torch.empty((15000, 1024, 1024), device="npu")
>>> del x
>>> torch_npu.npu.empty_virt_addr_cache()
>>> print(torch_npu.npu.memory_summary())
```
