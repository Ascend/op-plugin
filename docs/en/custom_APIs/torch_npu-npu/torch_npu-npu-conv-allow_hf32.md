# torch_npu.npu.conv.allow_hf32

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

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

Enables the capability to support the HF32 data type for convolution operators.

The functionality and calling method of `torch_npu.npu.conv.allow_hf32` are similar to those of `torch.backends.cudnn.allow_tf32`. For details about `torch.backends.cudnn.allow_tf32`, see [https://pytorch.org/docs/stable/backends.html\#torch.backends.cudnn.allow_tf32](https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.allow_tf32).

## Prototype

```python
torch_npu.npu.conv.allow_hf32: bool
```

## Parameters

Boolean configuration attribute with no return value. Assign a value directly to modify it. The default value is `True`.

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
