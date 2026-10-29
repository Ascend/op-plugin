# torch_npu.profiler.dynamic_profile.start

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

Triggers a `dynamic_profile` data collection.

## Prototype

```python
torch_npu.profiler.dynamic_profile.start(config_path: str = None)
```

## Parameters

**`config_path`** (`str`): Optional. Path to the `profiler_config.json` file. You must manually create the `profiler_config.json` configuration file and configure parameters as needed. A specific file name must be specified, such as `start("./home/xx/start_config_path/profiler_config.json")`. The paths `profiler_config_path` and `start_config_path` can contain only letters, digits, underscores, and hyphens. Symbolic links are not supported.

For details about the `profiler_config.json` file, see "<a href="https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/developer_notes/ascend_pytorch_profiler_user_guide.md">Ascend PyTorch Profiler</a>".

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
# Load the dynamic_profile module
from torch_npu.profiler import dynamic_profile as dp
# Set the profiling configuration file path for the init API
dp.init("profiler_config_path")
...
for step in steps:
    if step==5:
        # Set the profiling configuration file path for the start API
        dp.start("start_config_path")
    train_one_step()
    # Divide a step, where the code to be profiled must be placed between the dp.start() and dp.step() interfaces
    dp.step()
```
