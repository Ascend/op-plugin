# torch.npu.get_device_limit

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

- Obtains the device resource limits on a specified device.
- Currently, the supported resource types are Cube Core and Vector Core.

## Prototype

```python
torch.npu.get_device_limit(device) ->Dict
```

## Parameters

**`device`** (`Device`): Required. Device ID for resource limit control.

## Return Values

`Dict`

Number of Cube and Vector cores for the `device`.

## Constraints

None

## Example

 ```python
>>> import torch
>>> import torch_npu

>>> torch.npu.set_device(0)
>>> torch.npu.set_device_limit(0,12,20)
>>> print(torch.npu.get_device_limit(0))
{"cube_core_num":12, "vector_core_num":20}
 ```
