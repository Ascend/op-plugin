# torch_npu.profiler.ProfilerActivity

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Defines the event collection list, Enum type. It is used to assign a value to the `activities` parameter of `torch_npu.profiler.profile`.

## Prototype

```python
torch_npu.profiler.ProfilerActivity
```

## Parameters

- **`torch_npu.profiler.ProfilerActivity.CPU`**: Optional. Switch for framework-side data collection.
- **`torch_npu.profiler.ProfilerActivity.NPU`**: Optional. Switch for CANN software stack and NPU data collection.

By default, both of them are specified.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

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
