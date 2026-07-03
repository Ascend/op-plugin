# torch.npu.reset_stream_limit

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Resets the device resource limit of a specified stream to the default configuration after the `torch.npu.set_stream_limit` API has been used to set a custom device resource limit for that stream. The default resource settings can then be queried by calling the `torch.npu.get_stream_limit` API.

## Prototype

```python
torch.npu.reset_stream_limit(stream) -> None
```

## Parameters

**`stream`** (`torch_npu.npu.Stream`): Required. Stream for resource limit control.

## Return Values

`None`

No value is returned.

## Constraints

None

## Example

 ```python
>>> import torch
>>> import torch_npu

>>> torch.npu.set_stream_limit(torch.npu.current_stream(), 12, 24)
>>> torch.npu.reset_stream_limit(torch.npu.Stream())
 ```
