# set_custom_trace_id_callback

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Sets the callback for generating `trace_id`.

You can configure this callback directly using the `custom_trace_id_callback` parameter of the [torch_npu.profiler.profile](torch_npu-profiler-profile.md) API. Using the `custom_trace_id_callback` parameter is recommended.

## Prototype

```python
set_custom_trace_id_callback(self, callback)
```

## Parameters

- **`callback`** (`Callable`): Required. Custom callback function.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

After profile data collection is complete, `trace_id` is written to the `profiler_metadata.json` file.

```python
import torch
import torch_npu
...

# Define a trace ID generator
class RepeatTraceIdGenerator:
    def __init__(self):
        self.repeat_count = 0    # Count from 0

    def __call__(self) -> str:
        # The count increments by 1 each time profile data collection is started.
        current_id = self.repeat_count
        self.repeat_count += 1
        return str(current_id)

# Create a trace ID generator
trace_id_gen = RepeatTraceIdGenerator()

if __name__ == "__main__":
    device = torch.device('npu:1')
    torch.npu.set_device(device)
    x0 =torch.rand(3, 4).npu()
    x1 =torch.rand(3, 4).npu()


    stream = torch.npu.current_stream()
    stream.synchronize()

    # Add basic profiling configuration parameters
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
            ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0, skip_first_wait=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
    )

    prof.set_custom_trace_id_callback(trace_id_gen)    # Set the callback for generating trace IDs.
    prof.start()    # Start profile data collection.
    for i in range(12):    # Training function
        add(x0, x1)    # Training function
        prof.step()    # Used together with schedule.
        print(f"step {i}: {prof.get_trace_id()}")
    prof.stop()    # Stop profile data collection.
```
