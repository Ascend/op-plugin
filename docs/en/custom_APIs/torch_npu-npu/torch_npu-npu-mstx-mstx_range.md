# torch_npu.npu.mstx.mstx_range

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

Provides a range decorator to collect the range execution duration of the decorated function.

## Prototype

```python
torch_npu.npu.mstx.mstx_range(msg: str='None', stream=None, domain: str='default')
```

## Parameters

- **`msg`** (`str`): Optional. String pointer carrying instrumentation message data. The default value is `None`.
- **`stream`** (`int`): Optional. Stream used to execute the instrumentation task. The default value is `None`.
  - If set to `None`, only instantaneous events on the host are marked.
  - If set to a valid stream, the instantaneous events on the host and device are marked.
- **`domain`** (`str`): Optional. Name of the domain where time segment events are marked. The default value is `default`, indicating the default domain. If omitted, the default domain is used.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

```python
import torch
import torch_npu

@torch_npu.npu.mstx.mstx_range("train_one_step")
def train_one_step(step, steps, train_loader, model, optimizer, criterion):
    # Function code
    pass

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
       
    for epoch in range(epochs):
        for step in range(steps):
            train_one_step(step, steps, train_loader, model, optimizer, criterion)
            prof.step()


```
