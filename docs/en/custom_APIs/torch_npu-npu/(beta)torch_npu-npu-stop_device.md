# (beta) torch_npu.npu.stop_device

> [!NOTICE]  
> This API is reserved and is not supported currently.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term> |   √   |
|<term>Atlas A2 training products</term> |   √   |

## Function

Stops computation on the corresponding device and clears any unexecuted operations. Any subsequent computation on this device will raise an error.

## Prototype

```python
torch_npu.npu.stop_device(device_id: int) -> int 
```

## Parameters

**`device_id`** (`int`): ID of the device to be processed. Ensure that the device is valid.

## Return Values

`int`

An `int` value indicating the execution result. Valid values are `0` (execution succeeded) and `1` (execution failed).

## Example

 ```python
>>> import torch
>>> import torch_npu  
>>> torch.npu.set_device(0) 
>>> torch_npu.npu.stop_device(0)
 ```
