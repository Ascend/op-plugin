# torch_npu.profiler.AiCMetrics

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

AI Core performance metric collection items. They are used as the `aic_metrics` parameter of the `_ExperimentalConfig` class.

## Class Signature

```python
torch_npu.profiler.AiCMetrics
```

## Members

For details about result data of the following collection items, see <a href="https://gitcode.com/Ascend/msprof/blob/26.1.0/docs/en/user_guide/profile_data_file_references.md#op_summary-operator-details">op_summary (Operator Details)</a> in the *msProf* documentation. The actual collection results prevail.

- **`torch_npu.profiler.AiCMetrics.AiCoreNone`**: Disables the collection of AI Core performance metrics.
- **`torch_npu.profiler.AiCMetrics.PipeUtilization`**: Execution duration percentage of computation units and data transfer units.
- **`torch_npu.profiler.AiCMetrics.ArithmeticUtilization`**: Percentage statistics of various computation metrics.
- **`torch_npu.profiler.AiCMetrics.Memory`**: Percentage of external memory read/write instructions.
- **`torch_npu.profiler.AiCMetrics.MemoryL0`**: Percentage of internal L0 memory read/write instructions.
- **`torch_npu.profiler.AiCMetrics.ResourceConflictRatio`**: Percentage of pipeline queue instructions.
- **`torch_npu.profiler.AiCMetrics.MemoryUB`**: Percentage of internal UB memory read/write instructions.
- **`torch_npu.profiler.AiCMetrics.L2Cache`**: Read/write cache hit counts and reallocation counts after cache misses.
- **`torch_npu.profiler.AiCMetrics.MemoryAccess`**: Memory access bandwidth data of operators on the core.

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
import torch
import torch_npu

...

experimental_config = torch_npu.profiler._ExperimentalConfig(
       aic_metrics = torch_npu.profiler.AiCMetrics.AiCoreNone
       )
with torch_npu.profiler.profile(
        on_trace_ready = torch_npu.profiler.tensorboard_trace_handler("./result"),
        experimental_config = experimental_config) as prof:
        for step in range(steps): # Training function
                 train_one_step() # Training function
                 prof.step()
```
