# torch_npu.npu.ExternalEvent().wait()

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

Blocks execution of a specified stream until the specified event completes. This API supports only scenarios where a single stream waits for a single event.

## Prototype

```python
torch_npu.npu.ExternalEvent().wait(stream) -> None
```

## Parameters

**`stream`** (`torch_npu.npu.Stream`): Required. Specifies the stream used to deliver the event wait task.

## Return Values

None

## Constraints

- This API is asynchronous. A successful call indicates only that the task has been delivered, not that it has completed.
- API call sequence: `torch_npu.npu.ExternalEvent().wait()` -> `torch_npu.npu.ExternalEvent().record()` or `torch_npu.npu.ExternalEvent().record()` -> `torch_npu.npu.ExternalEvent().wait()`.
- This API automatically resets the event, so there is no need to call `torch_npu.npu.ExternalEvent().reset()` manually.

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
