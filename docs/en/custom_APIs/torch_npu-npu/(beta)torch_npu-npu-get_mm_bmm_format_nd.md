# (beta) torch_npu.npu.get_mm_bmm_format_nd

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

Determines whether the `mm` and `bmm` operators in the `linear` module have the ND layout enabled, returning `True` if ND is enabled and `False` otherwise.

## Prototype

```python
torch_npu.npu.get_mm_bmm_format_nd()
```

## Return Values

`bool`

## Example

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.get_mm_bmm_format_nd()
True
```
