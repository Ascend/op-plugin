# (beta) torch_npu.npu.get_autocast_dtype

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Obtains the data types supported by the device in the `AMP` scenario. This `dtype` is specified by `torch_npu.npu.set_autocast_dtype`. The default value is `float16`.

## Prototype

```python
torch_npu.npu.get_autocast_dtype()
```

## Return Values

`torch.dtype`

## Example

```python
import torch
import torch_npu

current_dtype = torch_npu.npu.get_autocast_dtype()

```
