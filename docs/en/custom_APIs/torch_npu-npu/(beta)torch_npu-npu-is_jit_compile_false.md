# (beta) torch_npu.npu.is_jit_compile_false

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Checks whether JIT compilation mode is disabled, returning `True` if it is disabled and `False` otherwise.

## Prototype

```python
torch_npu.npu.is_jit_compile_false()
```

## Return Values

A `bool` value.

## Example

```python
import torch
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
print(torch_npu.npu.is_jit_compile_false())
True
```
