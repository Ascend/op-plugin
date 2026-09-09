# (beta) torch_npu.empty_with_format

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.empty` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Returns a tensor filled with uninitialized data.

## Prototype

```python
torch_npu.empty_with_format(size, dtype, layout, device, pin_memory, acl_format)
```

## Parameters

- **`size`** (`List[int]`): Integer sequence defining the shape of the output tensor. It can be a variable number of arguments or a collection such as a list or tuple.
- **`dtype`** (`torch.dtype`): Optional. Desired data type of the returned tensor. The default value is `None`. If set to `None`, the global default data type is used. For details, see `torch.set_default_tensor_type()`.
- **`layout`** (`torch.layout`): Optional. Desired layout of the returned tensor. The default value is `torch.strided`.
- **`device`** (`torch.device`): Optional. Desired device of the returned tensor. The default value is `None`.
- **`pin_memory`** (`bool`): Optional. The default value is `False`. If this parameter is specified, the returned tensor is allocated in pinned memory.
- **`acl_format`** (`int`): Desired memory layout format of the returned tensor. The default value is `2`.

## Example

```python
>>> torch_npu.empty_with_format((2, 3), dtype=torch.float32, device="npu")
tensor([[3.1415e-45, 0.0000e+00, 1.2345e+20],
        [9.8765e-10, 0.0000e+00, 1.1111e+30]], device='npu:0')
# Note: `empty_with_format` returns an uninitialized tensor. The values in the example are for illustrative purposes only and are not fixed.
```
