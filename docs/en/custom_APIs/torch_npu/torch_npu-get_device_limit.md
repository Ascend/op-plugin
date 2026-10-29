# torch.npu.get_device_limit

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->

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
