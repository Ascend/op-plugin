# (beta) torch_npu.npu.set_autocast_dtype

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Controls the data types supported by the device in the AMP scenario.

## Prototype

```python
torch_npu.npu.set_autocast_dtype(dtype)
```

## Parameters

 **`dtype`**: Data type.

## Example

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.set_autocast_dtype(torch.float16)
```
