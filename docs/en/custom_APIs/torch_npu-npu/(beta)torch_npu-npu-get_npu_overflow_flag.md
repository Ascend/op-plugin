# (beta) torch_npu.npu.get_npu_overflow_flag

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas training products</term>                                      |    √     |

## Function

Checks whether value overflow occurs during NPU computation.

## Prototype

```python
torch_npu.npu.get_npu_overflow_flag()
```

## Example

```python
import torch
import torch_npu
a = torch.Tensor([65535]).npu().half()
a = a + a
ret = torch_npu.npu.get_npu_overflow_flag()
```
