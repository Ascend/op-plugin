# (beta) torch_npu.npu.set_mm_bmm_format_nd

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Controls whether the `mm` and `bmm` operators in the `linear` module use the ND layout.

## Prototype

```python
torch_npu.npu.set_mm_bmm_format_nd(bool)
```

## Example

```python
import torch
import torch_npu
torch_npu.npu.set_mm_bmm_format_nd(True)
```
