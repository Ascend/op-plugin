# torch_npu.profiler.profile

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Collects profile data during PyTorch training.

## Prototype

```python
torch_npu.profiler.profile(activities=None, schedule=None, on_trace_ready=None, record_shapes=False, profile_memory=False, with_stack=False, with_modules, with_flops=False, experimental_config=None)
```

## Parameters

- **`activities`** (`enum`): Optional. CPU or NPU event collection list. For details about the values and meanings, see [torch_npu.profiler.ProfilerActivity](./torch_npu-profiler-ProfilerActivity.md).

- **`schedule`** (`callable`): Optional. Sets the action for each step. It is controlled by the [schedule class](./torch_npu-profiler-schedule.md). If omitted, no operation is performed.

- **`on_trace_ready`** (`callable`): Optional. Automatically executes an operation when collection ends. This parameter supports only operations of the [tensorboard_trace_handler](./torch_npu-profiler-tensorboard_trace_handler.md) function. If omitted, no operation is performed.

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
  > Collecting profile data in environments where glibc is earlier than 2.34 can trigger a known glibc bug ([ID: 19329](https://sourceware.org/bugzilla/show_bug.cgi?id=19329)). Upgrading the environment glibc version resolves this issue.

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

- **`experimental_config`**: Optional. Configures common collection items of the performance profiling tool through extended configurations. For details about the supported collection items, see [torch_npu.profiler._ExperimentalConfig](./torch_npu-profiler-_ExperimentalConfig.md).

## Return Values

None

## Examples

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

The performance data collected by `torch_npu.profiler.profile` is automatically parsed and saved to the directory specified by `torch_npu.profiler.tensorboard_trace_handler`. Please refer to [MindStudio Insight System Tuning](https://gitcode.com/Ascend/msinsight/blob/26.0.0/docs/en/user_guide/system_tuning.md) for data visualization and analysis.

```python
import torch
import torch_npu

...

# Add extended configuration parameters for profiling. For details, see the following parameter description.
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone
)

# Add basic configuration parameters for profiling. For details, see the following parameter description.
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
        ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0, skip_first_wait=0),    # Used with prof.step()
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    profile_memory=False,
    with_modules=False,
    experimental_config=experimental_config) as prof:

    # Start profile data collection
    for step in range(steps):    # Training function
        train_one_step()    # Training function
        prof.step()    # Used with schedule
```
