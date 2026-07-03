# torch_npu.profiler.ProfilerAction

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Controls the profiler action state, Enum type. It manages actions such as profile data collection, profile data collection warmup, and profile data collection and saving.

## Prototype

```python
torch_npu.profiler.ProfilerAction
```

## Parameters

- **`torch_npu.profiler.ProfilerAction.NONE`**: Optional. No action is performed.
- **`torch_npu.profiler.ProfilerAction.WARMUP`**: Optional. Profile data collection warmup.
- **`torch_npu.profiler.ProfilerAction.RECORD`**: Optional. Profile data collection.
- **`torch_npu.profiler.ProfilerAction.RECORD_AND_SAVE`**: Optional. Profile data collection and saving.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

```python
import torch
import torch_npu

...
with torch_npu.profiler.profile(
    schedule=torch_npu.profiler.ProfilerAction.RECORD,
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
    ) as prof:
            for step in range(steps): # Training function
                train_one_step() # Training function
                prof.step()
```
