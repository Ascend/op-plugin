# torch_npu.profiler.schedule

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Sets the action for each step. It constructs the `schedule` parameter of `torch_npu.profiler.profile`. If omitted, no schedule operation is performed.

## Prototype

```python
torch_npu.profiler.schedule (wait, active, warmup = 0, repeat = 0, skip_first = 0, skip_first_wait = 0)
```

## Parameters

- **`wait`** (`int`): Required. Number of steps skipped before each repeated collection execution.

- **`active`** (`int`): Required. Number of profile data collection steps.

- **`warmup`** (`int`): Optional. Number of warmup steps. The default value is `0`. The recommended value is ``1`.

- **`repeat`** (`int`): Optional. Number of times to repeat the `wait + warmup + active` process. The value must be an integer greater than or equal to 0. The default value is `0`.

  When using a cluster analysis tool or MindStudio Insight for visualization, you are advised to set `repeat = 1`, which means the process is executed once and only one set of profile data is generated. This configuration is recommended for the following reasons:

  - If `repeat` is greater than 1, multiple sets of profile data are generated in the same directory. You must manually divide the collected profile data folders into `repeat` equal groups and place them into separate folders for parsing. The grouping must follow the chronological order of the timestamps in the folder names.
  - If `repeat` is set to `0`, the specific number of repeated executions is determined by the total training steps. For example, if total training steps is 100, `wait + active + warmup = 10`, and `skip_first = 10`, then `repeat = (100 - 10) / 10 = 9`. This indicates repeating 9 times and generating 9 sets of profile data.

- **`skip_first`** (`int`): Optional. Number of steps skipped before starting profile data collection. The default value is `0`. In dynamic shape scenarios, you are advised to skip the first 10 steps to ensure profile data stability. In other scenarios, you can configure this parameter as needed.

- **`skip_first_wait`** (`int`): Optional. Specifies whether to skip the first wait during profile data collection. The default value is `0`, which disables this parameter function. A non-zero integer enables this parameter function.<br>When this parameter is enabled, this function cancels the first wait of the `wait` parameter. Execution proceeds directly to the next action during the first step of the first repeat, but subsequent actions execute normally after the second repeat. This can save collection time.

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
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    schedule=torch_npu.profiler.schedule(
        wait=1,                        # Wait phase, skip 1 step
        warmup=1,                      # Warmup phase, skip 1 step
        active=2,                      # Record 2 steps of active data, and call on_trace_ready after execution
        repeat=2,                      # Repeat the wait + warmup + active process twice
        skip_first=1,                  # Skip 1 step
        skip_first_wait=1            # Skip the first wait
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./result')
    ) as prof:
        for _ in range(9):
            train_one_step()
            prof.step()                # Notify the profiler that 1 step is complete
```
