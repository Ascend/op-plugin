# (beta)torch_npu.contrib.module.npu_modules.DropoutWithByteMask

> [!NOTICE]
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Function

Applies an NPU-compatible `DropoutWithByteMask` operation.

## Prototype

```python
torch_npu.contrib.module.npu_modules.DropoutWithByteMask(p=0.5, inplace=False, max_seed=2 ** 10 - 1)
```

## Parameters

**Computation Parameters**

- **`p`** (`float`): Probability that elements are zeroed. The default value is `0.5`.
- **`inplace`** (`bool`): Specifies whether to perform the operation in-place. The default value is `False`.
- **`max_seed`**: Reserved parameter. Not used currently.

**Computation Input**

- **`input`** (`Tensor`): Input tensor. Any shape is supported.

## Return Values

`Tensor`

Output tensor with the same shape as the input tensor.

## Example

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module.npu_modules import DropoutWithByteMask
>>> m = DropoutWithByteMask(p=0.5)
>>> input = torch.randn(16, 16).npu()
>>> output = m(input)
>>> print(output.shape)
torch.Size([16, 16])
```
