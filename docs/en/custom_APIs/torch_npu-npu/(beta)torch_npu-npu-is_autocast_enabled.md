# (beta) torch_npu.npu.is_autocast_enabled

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Checks whether autocast is available.

## Prototype

```python
torch_npu.npu.is_autocast_enabled()
```

## Return Values

`bool`

## Example

``` python
import torch
import torch_npu
torch_npu.npu.is_autocast_enabled()
```
