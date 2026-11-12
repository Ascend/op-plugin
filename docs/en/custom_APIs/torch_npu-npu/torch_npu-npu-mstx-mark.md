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
torch_npu.npu.mstx.mark(message: str, stream=None, domain: str='default') -> None
```

## Parameters

- **`message`** (`str`): Required. String carrying information for the instrumentation point.

  Length of the `message` string must not exceed 255 bytes in msPTI scenarios.

- **`stream`** (`torch_npu.npu.Stream`): Optional. Stream used to execute the instrumentation task. The default value is `None`.

  - If set to `None`, only instantaneous events on the host are marked.
  - If set to a valid stream, the instantaneous events on the host and device are marked.

- **`domain`** (`str`): Optional. Name of the domain where instantaneous events are marked. The default value is `'default'`, indicating the default domain. If omitted, the default domain is used.

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
import torch
import torch_npu
import time
experimental_config = torch_npu.profiler._ExperimentalConfig(
    data_simplification=False,
    # Enable mstx and set the domain filter through mstx_domain_include or mstx_domain_exclude.
    mstx=True,
    mstx_domain_include=['default','domain1']    # Collect instrumentation data from 'default' and 'domain1'.
    # mstx_domain_exclude=['domain2']    # Do not collect instrumentation data from 'domain2'. Do not set it together with mstx_domain_include.
)
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0, skip_first_wait=0),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    experimental_config=experimental_config) as prof:
    for i in range(5):
        # Mark an mstx instrumentation point in the default domain.
        torch_npu.npu.mstx.mark("mark_with_default_domain")
        range_id = torch_npu.npu.mstx.range_start("range_with_default_domain")
        time.sleep(1)    # Simulate user code.
        torch_npu.npu.mstx.range_end(range_id)
        ...    # User code.
        # Mark an mstx instrumentation point in the user-defined domain1.
        torch_npu.npu.mstx.mark("mark_with_domain1", domain = "domain1")
        range_id1 = torch_npu.npu.mstx.range_start("range_with_domain1", domain="domain1")
        time.sleep(1)    # Simulate user code.
        torch_npu.npu.mstx.range_end(range_id1, domain="domain1")
        ...    # User code.
        # Mark an mstx instrumentation point in the user-defined domain2.
        torch_npu.npu.mstx.mark("mark_with_domain2", domain = "domain2")
        range_id2 = torch_npu.npu.mstx.range_start("range_with_domain2", domain="domain2")
        time.sleep(1)    # Simulate user code.
        torch_npu.npu.mstx.range_end(range_id2, domain="domain2")
        prof.step()

```
