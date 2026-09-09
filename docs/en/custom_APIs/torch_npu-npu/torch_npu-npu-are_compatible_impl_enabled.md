# torch_npu.npu.are_compatible_impl_enabled

## Supported Products

| Product                                                     | Supported|
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 training products</term>                       |    √     |
| <term>Atlas A3 inference products</term>                       |    √     |
| <term>Atlas A2 training products</term>                       |    √     |
| <term>Atlas A2 inference products</term>                       |    √     |
| <term>Atlas inference products</term>                          |    √     |
| <term>Atlas training products</term>                          |    √     |

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
