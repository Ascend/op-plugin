# torch_npu.npu.conv.allow_hf32

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Enables the capability to support the HF32 data type for convolution operators.

The functionality and calling method of `torch_npu.npu.conv.allow_hf32` are similar to those of `torch.backends.cudnn.allow_tf32`. For details about `torch.backends.cudnn.allow_tf32`, see [https://pytorch.org/docs/stable/backends.html\#torch.backends.cudnn.allow_tf32](https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.allow_tf32).

## Prototype

```python
torch_npu.npu.conv.allow_hf32 = bool
```

## Parameters

The input is a bool value. The default value is `True`.

## Output Description

`bool`

## Example

```python
>>>import torch
>>>import torch_npu
>>>print(torch_npu.npu.conv.allow_hf32)
True
>>>torch_npu.npu.conv.allow_hf32=False
>>>print(torch_npu.npu.conv.allow_hf32)
False
>>>torch_npu.npu.conv.allow_hf32=True
>>>print(torch_npu.npu.conv.allow_hf32)
True
```
