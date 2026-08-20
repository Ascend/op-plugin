# (beta) torch_npu.npu.restart_device

> [!NOTICE]  
> This API is reserved and is not supported currently.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term> |   √   |
|<term>Atlas A2 training products</term> |   √   |

## Function

Restores the state of the corresponding device so that subsequent computations on the device can continue normally.

## Prototype

```python
torch_npu.npu.restart_device(device_id, rebuild_all_resources=False, disable_tensor_unsafe_check=False) -> None
```

## Parameters

- **`device_id`** (`int`): Required. Device ID to be processed.
- **`rebuild_all_resources`** (`bool`): Optional. Specifies whether to rebuild resources on the device (default value: `False`). When set to `True`, all NPU streams on the device are restored. This parameter works with `disable_tensor_unsafe_check` to determine whether data is marked as dirty.
- **`disable_tensor_unsafe_check`** (`bool`): Optional. Specifies whether to disable dirty marking (default value: `False`). This parameter takes effect only when `rebuild_all_resources=True`. When set to `False`, all NPU tensors on the device are marked as dirty. When set to `True`, dirty marking is skipped.

## Constraints

The specified device ID must be valid, regardless of whether the device has been stopped.

## Example

 ```python
>>> import torch
>>> import torch_npu  
>>> torch.npu.set_device(0) 
>>> torch_npu.npu.stop_device(0)
>>> torch_npu.npu.restart_device(0)
 ```
