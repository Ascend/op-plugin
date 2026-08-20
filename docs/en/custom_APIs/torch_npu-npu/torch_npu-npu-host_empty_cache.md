# torch_npu.npu.host_empty_cache

## Supported Products

| Product                                                       | Supported|
| ----------------------------------------------------------- | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A3 inference products</term>  | √  |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas A2 inference products</term>|    √     |
|<term>Atlas inference products</term>|    √     |
|<term>Atlas training products</term>|    √     |

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
