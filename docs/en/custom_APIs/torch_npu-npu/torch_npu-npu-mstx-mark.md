# torch_npu.npu.mstx.mark

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

Marks an instantaneous event.

## Prototype

```python
torch_npu.npu.mstx.mark(message: str='None', stream=None, domain: str='default') -> none:
```

## Parameters

- **`message`** (`str`): Optional. String pointer carrying instrumentation message data. The default value is `None`. The length of the passed message string must satisfy the following constraints:
  - MSPTI scenarios: No more than 255 bytes.
  - Non-MSPTI scenarios: No more than 156 bytes.
- **`stream`** (`int`): Optional. Stream used to execute the instrumentation task. The default value is `None`.
  - If set to `None`, only instantaneous events on the host are marked.
  - If set to a valid stream, the instantaneous events on the host and device are marked.
- **`domain`** (`str`): Optional. Name of the domain where instantaneous events are marked. The default value is `default`, indicating the default domain. If omitted, the default domain is used.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

```python
import torch
import torch_npu

experimental_config = torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level_none,
    mstx=True,
    export_type=[
        torch_npu.profiler.ExportType.Db
        ])
with torch_npu.profiler.profile(
    schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=2, repeat=1, skip_first=1),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    experimental_config=experimental_config) as prof:
       
    for step in range(steps):
        train_one_step()    # User code that includes mstx API calls
        prof.step()
```
