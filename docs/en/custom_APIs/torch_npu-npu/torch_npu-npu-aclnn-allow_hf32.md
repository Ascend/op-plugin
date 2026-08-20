# torch_npu.npu.aclnn.allow_hf32

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Sets or queries whether convolution operators support HF32.

## Prototype

```python
torch_npu.npu.aclnn.allow_hf32:bool
```

## Parameters

**`bool`**: Specifies whether to enable or disable support for HF32.

## Return Values

A boolean indicator indicating whether `allow_hf32` is enabled. The default value is `True`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> res = torch_npu.npu.aclnn.allow_hf32
>>> print(res)
True
>>> torch_npu.npu.aclnn.allow_hf32 = False
>>> res = torch_npu.npu.aclnn.allow_hf32
>>> print(res)
False
>>> torch_npu.npu.aclnn.allow_hf32 = True
>>> res = torch_npu.npu.aclnn.allow_hf32
>>> print(res)
True
```
