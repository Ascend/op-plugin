# torch_npu.npu.ExternalEvent().reset()

> [!NOTICE]  
> This legacy API is scheduled for deprecation. `torch_npu.npu.ExternalEvent().wait()` automatically resets the event, which eliminates the requirement to call this API to manually reset the event.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A3 inference products</term>  | √  |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas A2 inference products</term>|    √     |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

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
- API core sequence: `torch_npu.npu.ExternalEvent().wait()` -> `torch_npu.npu.ExternalEvent().reset()` -> `torch_npu.npu.ExternalEvent().record()` or `torch_npu.npu.ExternalEvent().record()` -> `torch_npu.npu.ExternalEvent().wait()` -> `torch_npu.npu.ExternalEvent().reset()`.

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
