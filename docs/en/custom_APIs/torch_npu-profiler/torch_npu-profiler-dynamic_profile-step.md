# torch_npu.profiler.dynamic_profile.step

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

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

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
