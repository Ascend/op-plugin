# torch_npu.profiler.profile.enable_profiler_in_child_thread

## Supported Products

| Product                                                     | Supported|
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 training products</term>                       |    √     |
| <term>Atlas A3 inference products</term>                       |    √     |
| <term>Atlas A2 training products</term>                       |    √     |
| <term>Atlas A2 inference products</term>|    √     |
| <term>Atlas inference products</term>                          |    √     |
| <term>Atlas training products</term>                          |    √     |

## Function

Registers a profile data collection callback function. This function collects framework-side data such as torch operators launched under user child threads. Parameters of [torch_npu.profiler.profile](./torch_npu-profiler-profile.md) (such as `record_shapes`, `profile_memory`, `with_stack`, `with_flops`, and `with_modules`) can also be configured as collection settings for child threads.

Use it together with [torch_npu.profiler.profile.disable_profiler_in_child_thread](./torch_npu-profiler-profile-disable_profiler_in_child_thread.md).

## Prototype

```python
torch_npu.profiler.profile.enable_profiler_in_child_thread(record_shapes=False, profile_memory=False, with_stack=False, with_modules=False, with_flops=False)
```

## Parameters

- **`record_shapes`** (`bool`): Optional. Specifies whether to record input shapes and input types of operators. Valid values are:

  - `True`: enabled
  - `False`: disabled

  The default value is `False`.

  This parameter takes effect when `torch_npu.profiler.ProfilerActivity.CPU` is enabled.

- **`profile_memory`** (`bool`): Optional. Specifies whether to record memory allocation status of operators. Valid values are:

  - `True`: enabled
  - `False`: disabled

  The default value is `False`.

  > [!NOTE]  
  > Collecting profile data in environments where glibc is earlier than 2.34 can trigger a known glibc bug ([ID: 19329](https://sourceware.org)). Upgrading the environment glibc version resolves this issue.

- **`with_stack`** (`bool`): Optional. Specifies whether to record call stack information. This includes call information of the framework layer or CPU operator layer. Valid values are:

  - `True`: enabled
  - `False`: disabled

  The default value is `False`.

  This parameter takes effect when `torch_npu.profiler.ProfilerActivity.CPU` is enabled.

- **`with_modules`** (`bool`): Optional. Specifies whether to record the module-level Python call stack, which is call information at the framework layer. Valid values are:

  - `True`: enabled
  - `False`: disabled

  The default value is `False`.

  This parameter takes effect when `torch_npu.profiler.ProfilerActivity.CPU` is enabled.

- **`with_flops`** (`bool`): Optional. Specifies whether to record floating-point operations of operators. Profile data analysis for this parameter is not supported. Valid values are:

  - `True`: enabled
  - `False`: disabled

  The default value is `False`.

  This parameter takes effect when `torch_npu.profiler.ProfilerActivity.CPU` is enabled.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

```python
import threading
import random
import torch
import torch_npu

# Define the inference model
...

def infer(device, child_thread):
    torch.npu.set_device(device)

    if child_thread:
        # Start collecting framework-side data such as torch operators from the child thread
        torch_npu.profiler.profile.enable_profiler_in_child_thread(with_modules=True)
    
    for _ in range(5):
        outputs = model(input_data)

    if child_thread:
        # Stop collecting framework-side data such as torch operators from the child thread
        torch_npu.profiler.profile.disable_profiler_in_child_thread()


if __name__ == "__main__":
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    )

    prof = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_multi", analyse_flag=True),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=True,
        experimental_config=experimental_config)

    prof.start()

    threads = []
    for i in range(1, 3):
        # Create two child threads to perform inference tasks on device1 and device2, respectively
        t = threading.Thread(target=infer, args=(i, True))
        t.start()
        threads.append(t)

    # Run the inference task on device0 in the main thread
    infer(0, False)

    for t in threads:
        t.join()

    prof.stop()
```
