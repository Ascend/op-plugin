# (beta) torch_npu.contrib.module.NpuDropPath

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

Replaces the native `DropPath` implementation in `swin_transformer.py` with an NPU-optimized implementation. It randomly drops the main path (stochastic depth) of each sample within the residual blocks.

## Prototype

```python
torch_npu.contrib.module.NpuDropPath(drop_prob=None)
```

## Parameters

**Computation Parameters**

- **`drop_prob`** (`float`): DropPath probability (path dropping probability).

**Computation Input**

- **`x`** (`Tensor`): Input tensor to which DropPath is applied.

## Return Values

`Tensor`

DropPath computation result.

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
