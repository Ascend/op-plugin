# torch_npu.profiler.supported_profiler_level

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
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

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
