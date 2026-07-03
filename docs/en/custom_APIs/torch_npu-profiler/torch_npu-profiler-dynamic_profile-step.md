# torch_npu.profiler.dynamic_profile.step

## Supported Products

| Product                              | Supported|
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term>|    √     |
| <term>Atlas A2 training products</term>|    √     |
| <term>Atlas training products</term>   |    √     |

## Function

Divides steps during `dynamic_profile` data collection.

## Prototype

```python
torch_npu.profiler.dynamic_profile.step()
```

## Parameters

None

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
