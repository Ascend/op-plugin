# (beta) torch_npu.contrib.module.npu_modules.DropoutWithByteMask

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

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
