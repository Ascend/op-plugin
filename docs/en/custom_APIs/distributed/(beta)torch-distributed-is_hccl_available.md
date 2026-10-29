# (beta) torch.distributed.is_hccl_available

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

## Function

Determines whether the `HCCL` communication backend is available, similar to `torch.distributed.is_nccl_available`. For details, see [https://pytorch.org/docs/stable/distributed.html\#torch.distributed.is_nccl_available](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_nccl_available).

## Prototype

```python
torch.distributed.is_hccl_available()
```

## Return Values

A `bool` value: `True` (available) or `False` (unavailable).

## Example

```python
import torch
import torch_npu

print(torch.distributed.is_hccl_available())

True
```
