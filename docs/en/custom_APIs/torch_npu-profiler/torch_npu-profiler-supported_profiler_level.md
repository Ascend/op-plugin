# torch_npu.profiler.supported_profiler_level

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Queries the supported levels of `torch_npu.profiler.ProfilerLevel`.

## Prototype

```python
torch_npu.profiler.supported_profiler_level()
```

## Return Values

If {'Level0', 'Level1', 'Level_none', 'Level2'} is returned, the operation is successful. If nothing is returned, the operation fails.

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
import torch
import torch_npu

...

torch_npu.profiler.supported_profiler_level()
```
