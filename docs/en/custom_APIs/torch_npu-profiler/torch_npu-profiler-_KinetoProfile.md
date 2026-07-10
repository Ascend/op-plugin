# torch_npu.profiler._KinetoProfile

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
torch_npu.profiler._KinetoProfile(activities=None, record_shapes=False, profile_memory=False, with_stack=False, with_flops=False, with_modules=False, experimental_config=None)
```

## Parameters

- **`activities`** (`enum`): Optional. CPU or NPU event collection list. For details about the values and meanings, see [torch_npu.profiler.ProfilerActivity](torch_npu-profiler-ProfilerActivity.md).

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

- **`with_flops`** (`bool`): Optional. Specifies whether to record floating-point operations of operators. Profile data analysis for this parameter is not supported. Valid values are:

    - `True`: enabled
    - `False`: disabled

     The default value is `False`.

     This parameter takes effect when `torch_npu.profiler.ProfilerActivity.CPU` is enabled.

- **`experimental_config`**: Optional. Extended configuration for profile data collection. For details about the supported collection items, see [torch_npu.profiler._ExperimentalConfig](torch_npu-profiler-_ExperimentalConfig.md).

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

```python
import torch
import torch_npu

...

prof = torch_npu.profiler._KinetoProfile(activities=None, record_shapes=False, profile_memory=False, with_stack=False, with_flops=False, with_modules=False, experimental_config=None)
for epoch in range(epochs):
        train_model_step()
        if epoch == 0:
                prof.start()
        if epoch == 1:
                prof.stop()
prof.export_chrome_trace("result_dir/trace.json")
```
