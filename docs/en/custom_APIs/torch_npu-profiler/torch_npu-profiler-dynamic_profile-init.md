# torch_npu.profiler.dynamic_profile.init

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Initializes `dynamic_profile`.

## Prototype

```python
torch_npu.profiler.dynamic_profile.init(path: str)
```

## Parameters

**`path`** (`str`): Required. Specifies the path where `dynamic_profile` automatically creates the template file `profiler_config.json`. You can modify configuration items based on this template file. The path can contain only letters, digits, underscores, and hyphens. Symbolic links are not supported.

For details about the `profiler_config.json` file, see section "<a href="https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0033.html">Ascend PyTorch Profiler</a>" in *CANN Performance Tuning Tool*.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

```python
# Load the dynamic_profile module
from torch_npu.profiler import dynamic_profile as dp
# Set the path to the profiling configuration file
dp.init("profiler_config_path")
...
for step in steps:
    train_one_step()
    # Divide a step
    dp.step()
```
