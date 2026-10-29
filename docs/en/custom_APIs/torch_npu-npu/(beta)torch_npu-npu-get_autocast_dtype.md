# (beta) torch_npu.npu.get_autocast_dtype

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

Obtains the data types supported by the device in the `AMP` scenario. This `dtype` is specified by `torch_npu.npu.set_autocast_dtype`. If omitted, the default data type `torch.float16` is used.

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
