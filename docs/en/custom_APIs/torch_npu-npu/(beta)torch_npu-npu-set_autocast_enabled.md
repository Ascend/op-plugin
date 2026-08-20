# (beta) torch_npu.npu.set_autocast_enabled

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Controls whether to enable AMP on the device.

## Prototype

```python
torch_npu.npu.set_autocast_enabled(bool)
```

## Parameters

**`bool`**: Valid values are `True` (enables AMP on the device) or `False` (disables AMP).

## Example

```python
import torch
import torch_npu
torch_npu.npu.set_autocast_enabled(True)
```
