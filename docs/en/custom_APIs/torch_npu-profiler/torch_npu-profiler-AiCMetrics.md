# torch_npu.profiler.AiCMetrics

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Defines AI Core performance metric collection items (enumeration type), used as the `aic_metrics` parameter of the `_ExperimentalConfig` class.

## Prototype

```python
torch_npu.profiler.AiCMetrics
```

## Parameters

For details about result data of the following collection items, see <a href="https://www.hiascend.com/document/detail/en/canncommercial/900/devaids/Profiling/atlasprofiling_16_0067.html">op_summary (Operator Details)</a> in *CANN Profiling*. The actual collection results prevail.

- **`torch_npu.profiler.AiCMetrics.AiCoreNone`**: Optional. Disables the collection of AI Core performance metrics.
- **`torch_npu.profiler.AiCMetrics.PipeUtilization`**: Optional. Execution duration percentage of computation units and data transfer units.
- **`torch_npu.profiler.AiCMetrics.ArithmeticUtilization`**: Optional. Percentage statistics of various computation metrics.
- **`torch_npu.profiler.AiCMetrics.Memory`**: Optional. Percentage of external memory read/write instructions.
- **`torch_npu.profiler.AiCMetrics.MemoryL0`**: Optional. Percentage of internal L0 memory read/write instructions.
- **`torch_npu.profiler.AiCMetrics.ResourceConflictRatio`**: Optional. Percentage of pipeline queue instructions.
- **`torch_npu.profiler.AiCMetrics.MemoryUB`**: Optional. Percentage of internal UB memory read/write instructions.
- **`torch_npu.profiler.AiCMetrics.L2Cache`**: Optional. Read/write cache hit counts and reallocation counts after cache misses.
- **`torch_npu.profiler.AiCMetrics.MemoryAccess`**: Optional. Memory access bandwidth data of operators on the core.

The default value is `torch_npu.profiler.AiCMetrics.AiCoreNone`.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

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
