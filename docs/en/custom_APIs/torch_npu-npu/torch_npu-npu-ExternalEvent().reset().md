# torch_npu.npu.ExternalEvent().reset()

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.<br>
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

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

Resets an event. In event reuse scenarios, this API resets the flag bit set after the `record` operation completes.

## Prototype

```python
torch_npu.npu.ExternalEvent().reset(stream) -> None
```

## Parameters

**`stream`** (`torch_npu.npu.Stream`): Required. Specifies the stream used to deliver the event reset task.

## Return Values

None

## Constraints

- This API is asynchronous. A successful call indicates only that the task has been delivered, not that it has completed.
- API call sequence: `torch_npu.npu.ExternalEvent().wait()` -> `torch_npu.npu.ExternalEvent().reset()` -> `torch_npu.npu.ExternalEvent().record()` or `torch_npu.npu.ExternalEvent().record()` -> `torch_npu.npu.ExternalEvent().wait()` -> `torch_npu.npu.ExternalEvent().reset()`.

## Example

```python
import torch
import torch_npu

torch.npu.set_device(0)

event = torch.npu.ExternalEvent()
default_stream = torch_npu.npu.current_stream()
stream = torch.npu.Stream()

event.wait(default_stream)
event.reset(default_stream)
event.record(stream)
```
