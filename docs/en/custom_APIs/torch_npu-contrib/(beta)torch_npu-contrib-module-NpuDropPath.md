# (beta) torch_npu.contrib.module.NpuDropPath

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Replaces the native `DropPath` implementation in `swin_transformer.py` with an NPU-optimized implementation. It randomly drops the main path (stochastic depth) of each sample within the residual blocks.

## Prototype

```python
torch_npu.contrib.module.NpuDropPath(drop_prob=None)
```

## Parameters

**Computation Parameters**

- **`drop_prob`** (`float`): Dropout probability.

**Computation Input**

- **`x`** (`Tensor`): Input tensor to which DropPath is applied.

## Return Values

`Tensor`

Dropout computation result.

## Example

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module import NpuDropPath
>>> input1 = torch.randn(68, 5).npu()
>>> input1.requires_grad_(True)
>>> input2 = torch.randn(68, 5).npu()
>>> input2.requires_grad_(True)
>>> fast_drop_path = NpuDropPath(0).npu()
>>> output = input1 + fast_drop_path(input2)
>>> output.sum().backward()
```
