# torch_npu.profiler.ProfilerAction

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Profiler collection action state, enum type. It is returned by `torch_npu.profiler.schedule` at each step and controls the collection action of `torch_npu.profiler.profile` for that step: no action, warmup, collection, or collection and saving.

## Class Signature

```python
class ProfilerAction(Enum)
```

## Members

> All enum values are read-only and cannot be modified at runtime.

| Member | Value | Description |
| :--- | :--- | :--- |
| `NONE` | `0` | No action is performed. |
| `WARMUP` | `1` | Warms up profile data collection. |
| `RECORD` | `2` | Collects profile data. |
| `RECORD_AND_SAVE` | `3` | Collects and saves profile data. |

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

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
