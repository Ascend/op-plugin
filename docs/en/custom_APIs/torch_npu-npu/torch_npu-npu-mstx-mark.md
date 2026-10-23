# torch_npu.npu.mstx.mark

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 inference products</term>: Supported
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas inference products</term>: Supported
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas training products</term>: Supported
<!-- end id7 -->

## Function

Marks an instantaneous event.

## Prototype

```python
torch_npu.npu.mstx.mark(message: str, stream=None, domain: str='default') -> None
```

## Parameters

- **`message`** (`str`): Optional. String carrying information for the instrumentation point (default value: `'None'`).

  Requirement on the length of the passed message string: in the msPTI scenario, it must not exceed 255 bytes.

- **`stream`** (`torch_npu.npu.Stream`): Optional. Stream used to execute the instrumentation task (default value: `None`).
  
  - When set to None, only the instantaneous event on the host side is marked.
  - When set to a valid stream, the instantaneous events on the host side and the corresponding device side are marked.
  
- **`domain`** (`str`): Optional. Name of the domain where instantaneous events are marked. The default value is 'default', indicating the default domain. If omitted, the default domain is used.

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
    # Enable mstx and set the domain filter attribute of mstx_domain_include or mstx_domain_exclude
    mstx=True,
    mstx_domain_include=['default','domain1']    # Configure the collection of point data for 'default' and 'domain1'
    # mstx_domain_exclude=['domain2']    # Configure not to collect the point data of 'domain2'. This cannot be configured at the same time as mstx_domain_include.
)
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0, skip_first_wait=0),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    experimental_config=experimental_config) as prof:
    for i in range(5):
        # Mark the mstx point of the default domain
        torch_npu.npu.mstx.mark("mark_with_default_domain")
        range_id = torch_npu.npu.mstx.range_start("range_with_default_domain")
        time.sleep(1)    # Simulate user code
        torch_npu.npu.mstx.range_end(range_id)
        ...    # User code
        # Mark the mstx point of the user-defined domain1
        torch_npu.npu.mstx.mark("mark_with_domain1", domain = "domain1")
        range_id1 = torch_npu.npu.mstx.range_start("range_with_domain1", domain="domain1")
        time.sleep(1)    # Simulate user code
        torch_npu.npu.mstx.range_end(range_id1, domain="domain1")
        ...    # User code
        # Mark the mstx point of the user-defined domain2
        torch_npu.npu.mstx.mark("mark_with_domain2", domain = "domain2")
        range_id2 = torch_npu.npu.mstx.range_start("range_with_domain2", domain="domain2")
        time.sleep(1)    # Simulate user code
        torch_npu.npu.mstx.range_end(range_id2, domain="domain2")
        prof.step()

```
