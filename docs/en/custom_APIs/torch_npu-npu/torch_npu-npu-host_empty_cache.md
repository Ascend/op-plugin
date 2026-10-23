# torch_npu.npu.host_empty_cache

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 inference products</term>: Supported
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas inference products</term>: Supported
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas training products</term>: Supported
<!-- end id7 -->

## Function

Releases all unused host physical memory currently held by the cache.

## Definition File

torch_npu/npu/memory.py

## Prototype

```python
torch_npu.npu.host_empty_cache()
```

## Parameters

None

## Return Values

None

## Constraints

None

## Example

```python
>>> import torch
>>> import torch_npu
>>> x = torch.empty([1024, 1024]).pin_memory()
>>> del x
>>> torch_npu.npu.host_empty_cache()
>>> print(torch_npu.npu.host_memory_stats())
```
