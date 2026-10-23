# torch_npu.npu.ExternalEvent

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

Wraps the AscendCL Event. During graph capture in `NPUGraph` scenarios, `ExternalEvent` is captured as an external graph node for timing control outside the graph.

## Prototype

```python
torch_npu.npu.ExternalEvent()
```

## Return Values

The created `ExternalEvent` object, which is used to deliver event-related tasks.

## Constraints

Creating an `ExternalEvent` allocates 32 bytes of device memory. The number of `ExternalEvent` objects is limited by chip hardware specifications.

## Example

```python
import torch
import torch_npu

torch.npu.set_device(0)

event = torch_npu.npu.ExternalEvent()
default_stream = torch_npu.npu.current_stream()
stream = torch.npu.Stream()

event.wait(default_stream)
event.record(stream)
```
