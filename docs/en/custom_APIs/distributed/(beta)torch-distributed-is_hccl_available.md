# (beta) torch.distributed.is_hccl_available

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Determines whether the `HCCL` communication backend is available, similar to `torch.distributed.is_nccl_available`. For details, see [https://pytorch.org/docs/stable/distributed.html\#torch.distributed.is_nccl_available](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_nccl_available).

## Prototype

```python
torch.distributed.is_hccl_available()
```

## Return Values

A `Bool` value: `True` (available) or `False` (unavailable).

## Example

```python
import torch
import torch_npu

print(torch.distributed.is_hccl_available())

True
```
