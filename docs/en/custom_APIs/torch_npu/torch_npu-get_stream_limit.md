# torch.npu.get_stream_limit

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

- Obtains the device resource limits of a specified stream.
- If the `torch.npu.set_stream_limit` API is not called to set device resource limits, the device resource limits obtained using this API are determined based on the following priority: current process resource limits (set by calling the `torch.npu.set_device_limit` API) > hardware default resource limits.
- Currently, the supported resource types are Cube Core and Vector Core.

## Prototype

```python
torch.npu.get_stream_limit(stream) ->Dict
```

## Parameters

**`stream`** (`torch_npu.npu.Stream`): Required. Stream for resource limit control.

## Return Values

`Dict`

Number of Cube and Vector cores for the `stream`.

## Constraints

None

## Example

 ```python
>>> import torch
>>> import torch_npu

>>> torch.npu.set_stream_limit(torch.npu.current_stream(),12,20)
>>> print(torch.npu.get_stream_limit(torch.npu.current_stream()))
{"cube_core_num":12, "vector_core_num":20}
 ```
