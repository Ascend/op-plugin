# (beta) torch_npu.npu.set_aoe

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Enables AOE optimization.

## Prototype

```python
torch_npu.npu.set_aoe(dump_path)
```

## Parameters

**`dump_path`**: Path for saving the dumped operator graph.

## Example

```python
import torch
import torch_npu
import os

os.mkdir("./aoe_dump")
dump_path = "./aoe_dump"
torch_npu.npu.set_aoe(dump_path)

```
