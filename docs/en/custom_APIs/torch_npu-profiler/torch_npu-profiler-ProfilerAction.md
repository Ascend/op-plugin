# torch_npu.profiler.ProfilerAction

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

Profiler collection action status, an enumeration type. It is returned by `torch_npu.profiler.schedule` at each step and is used to control the collection action performed by `torch_npu.profiler.profile` at that step (no action, warmup, collection, or collection and save).

## Class Signature

```python
class ProfilerAction(Enum)
```

## Member Description

> All enumeration values are read-only and cannot be modified at runtime.

| Member | Value | Description |
| :--- | :--- | :--- |
| `NONE` | `0` | No action is performed. |
| `WARMUP` | `1` | Profile data collection warmup. |
| `RECORD` | `2` | Profile data collection. |
| `RECORD_AND_SAVE` | `3` | Profile data collection and saving. |

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
