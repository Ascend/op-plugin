# torch_npu.npu.empty_virt_addr_cache

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->

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
>>> import os
>>> import torch
>>> import torch_npu
>>> os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
>>> x = torch.empty((15000, 1024, 1024), device="npu")
>>> del x
>>> torch_npu.npu.empty_virt_addr_cache()
>>> print(torch_npu.npu.memory_summary())
```
