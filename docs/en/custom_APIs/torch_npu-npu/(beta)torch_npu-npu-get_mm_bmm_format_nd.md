# (beta) torch_npu.npu.get_mm_bmm_format_nd

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

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
