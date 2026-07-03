# torch_npu.npu.ExternalEvent

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

Encapsulates the AscendCL Event. During graph capture in `NPUGraph` scenarios, an `ExternalEvent` is captured as an external graph node and is used for timing control outside the graph.

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
