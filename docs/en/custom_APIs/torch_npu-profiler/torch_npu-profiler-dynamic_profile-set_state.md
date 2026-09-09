# torch_npu.profiler.dynamic_profile.set_state

## Supported Products

| Product                               | Supported |
| ---------------------------------- | :------: |
| <term>Atlas A3 training products</term> |    √     |
| <term>Atlas A2 training products</term> |    √     |
| <term>Atlas training products</term>    |    √     |

## Function

Sets the number of training steps executed during dynamic profiling.

When no training interruption occurs, dynamic profiling starts counting from step 0 and starts profiling when training reaches the corresponding step. If training is interrupted and then restarted, and this API is not configured, dynamic profiling by default treats the point at which training is restarted as step 0. As a result, profiling may not start before training ends. This API can be used to manually set the number of training steps already executed when training is interrupted, so that dynamic profiling starts counting from the specified step.

## Prototype

```python
torch_npu.profiler.dynamic_profile.set_state(state_step: dict)
```

## Parameters

**`state_step`** (`dict`): Optional. Sets the number of training steps already executed. The user must manually set the required value based on the actual training progress. The value must be an integer greater than or equal to `0`. The default value is `0`.

## Return Values

None.

## Examples

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
# Load the dynamic_profile module
from torch_npu.profiler import dynamic_profile as dp
# Set the number of training steps already executed
dp.set_state({"cur_step": 10})
dp.init("/data/test")
for t in range(50):
    if t <= 10:    # Simulate the number of training steps already executed when training is interrupted
       continue
    train_one_step()
    # Divide training into steps. The code to be profiled must be placed between dp.start() and dp.step().
    dp.step()
```
