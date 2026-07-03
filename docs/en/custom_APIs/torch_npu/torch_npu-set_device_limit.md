# torch.npu.set_device_limit

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Sets the number of Cube and Vector cores used by a specified device in a process during operator execution.

## Prototype

```python
torch.npu.set_device_limit(device, cube_num=-1, vector_num=-1) -> None
```

## Parameters

- **`device`** (`Device`): Required. ID of the target device.
- **`cube_num`** (`int`): Optional. Number of Cube cores. The default value is `-1` (no core splitting limit is configured).
- **`vector_num`** (`int`): Optional. Number of Vector cores. The default value is `-1` (no core splitting limit is configured).

## Return Values

`None`

No value is returned.

## Constraints

- This API can be called only once and takes effect only for the current process.
- You must call `set_device` before using this API.

## Example

 ```python
>>> import torch
>>> import torch_npu

>>> torch.npu.set_device(0)
>>> torch.npu.set_device_limit(0, 12, 24)
 ```
