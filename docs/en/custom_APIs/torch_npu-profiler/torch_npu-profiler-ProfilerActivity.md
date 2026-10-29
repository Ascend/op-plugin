# torch_npu.profiler.ProfilerActivity

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

Event collection list, an Enum type. It is used to assign a value to the activities parameter of torch_npu.profiler.profile.

## Class Signature

```python
torch_npu.profiler.ProfilerActivity
```

## Member Description

- **`torch_npu.profiler.ProfilerActivity.CPU`**: Optional. Switch for framework-side data collection.
- **`torch_npu.profiler.ProfilerActivity.NPU`**: Optional. Switch for CANN software stack and NPU data collection.

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
import torch
import torch_npu

...

with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
            ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
) as prof:
        for step in range(steps): # Training function
            train_one_step() # Training function
            prof.step()
```
