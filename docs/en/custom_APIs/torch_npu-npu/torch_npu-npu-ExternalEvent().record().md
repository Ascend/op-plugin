# torch_npu.npu.ExternalEvent().record()

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

Records an event on a specified stream. This API captures all tasks delivered on the current stream and records them in the event. As a result, a subsequent call to the `wait` API will wait until all tasks captured by the event have completed.

## Prototype

```python
torch_npu.npu.ExternalEvent().record(stream) -> None
```

## Parameters

 **`stream`** (`torch_npu.npu.Stream`): Required. Specifies the stream used to deliver the event recording task.

## Return Values

None

## Constraints

- This API is asynchronous. A successful call indicates only that the task has been delivered, not that it has completed.
- API call sequence: `torch_npu.npu.ExternalEvent().wait()` -> `torch_npu.npu.ExternalEvent().record()` or `torch_npu.npu.ExternalEvent().record()` -> `torch_npu.npu.ExternalEvent().wait()`.

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
