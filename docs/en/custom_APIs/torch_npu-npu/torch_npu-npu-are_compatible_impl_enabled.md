# torch_npu.npu.are_compatible_impl_enabled

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 inference products</term>: Supported
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas inference products</term>: Supported
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas training products</term>: Supported
<!-- end id7 -->

## Function

Queries the configuration of `torch_npu.npu.use_compatible_impl` to check whether operator APIs are fully aligned with the community implementations.

## Prototype

```python
torch_npu.npu.are_compatible_impl_enabled()
```

## Parameters

None

## Return Values

`bool`

`True` (enabled) or `False` (disabled).

## Constraints

None

## Example

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.use_compatible_impl(True)
>>> torch_npu.npu.are_compatible_impl_enabled()
True
```
