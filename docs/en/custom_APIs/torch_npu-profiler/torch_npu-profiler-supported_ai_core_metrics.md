# torch_npu.profiler.supported_ai_core_metrics

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

Queries the supported AI Core performance metric collection items of `torch_npu.profiler.AiCMetrics`.

## Prototype

```python
torch_npu.profiler.supported_ai_core_metrics()
```

## Return Values

If {'ACL_AICORE_MEMORY_ACCESS', 'ACL_AICORE_NONE', 'ACL_AICORE_L0B_AND_WIDTH', 'ACL_AICORE_L2_CACHE', 'ACL_AICORE_MEMORY_BANDWIDTH', 'ACL_AICORE_MEMORY_UB', 'ACL_AICORE_PIPE_UTILIZATION', 'ACL_AICORE_ARITHMETIC_UTILIZATION', 'ACL_AICORE_RESOURCE_CONFLICT_RATIO'} is returned, the operation is successful. If nothing is returned, the operation fails.

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
import torch
import torch_npu

...

torch_npu.profiler.supported_ai_core_metrics()
```
