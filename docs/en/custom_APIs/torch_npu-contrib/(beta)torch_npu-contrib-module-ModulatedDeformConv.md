# (beta) torch_npu.contrib.module.ModulatedDeformConv

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Applies an NPU-based modulated deformable 2D convolution operation.

## Prototype

```python
torch_npu.contrib.module.ModulatedDeformConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True, pack=True)
```

## Parameters

**Computation Parameters**

- **`in_channels`** (`int`): Number of channels in the input image.
- **`out_channels`** (`int`): Number of output channels produced by the convolution.
- **`kernel_size`** (`int`, `tuple`): Size of the convolution kernel.
- **`stride`** (`int`, `tuple`): Stride of the convolution. The default value is `1`.
- **`padding`** (`int`, `tuple`): Optional. Zero-padding added to both sides of the input. The default value is `0`.
- **`dilation`** (`int`, `tuple`): Spacing between convolution kernel elements. The default value is `1`.
- **`groups`** (`int`): Number of blocked connections from the input channels to the output channels. The default value is `1`.
- **`deformable_groups`** (`int`): Number of deformable groups.
- **`bias`** (`bool`): Specifies whether to add a learnable bias to the output. The default value is `False`.
- **`pack`** (`bool`): Optional. Specifies whether to include `conv_offset` and the mask in this module. The default value is `True`.

**Computation Input**

- **`x`** (`Tensor`): Input tensor.

## Return Values

`Tensor`

Convolution computation result.

## Constraints

`ModulatedDeformConv` supports only the `float32` data type. The weights and biases in `conv_offset` must be initialized to `0`.

## Example

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module import ModulatedDeformConv
>>> m = ModulatedDeformConv(32, 32, 1).npu()
>>> input_tensor = torch.randn(2, 32, 5, 5).npu()
>>> output = m(input_tensor)
>>> print(output.shape)
torch.Size([2, 32, 5, 5])
```
