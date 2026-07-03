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

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

```python
import torch
import torch_npu

...

torch_npu.profiler.supported_profiler_level()
```
