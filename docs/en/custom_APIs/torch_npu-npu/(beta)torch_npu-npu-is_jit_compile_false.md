# (beta) torch_npu.npu.is_jit_compile_false

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

Checks whether JIT compilation mode is disabled, returning `True` if it is disabled and `False` otherwise.

<!-- npu="950" id6 -->
For <term>Ascend 950DT products</term>, only `True` is returned, that is, the JIT compilation mode is disabled by default.
<!-- end id6 -->

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
