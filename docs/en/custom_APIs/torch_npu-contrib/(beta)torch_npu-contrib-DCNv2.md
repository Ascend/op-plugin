# (beta) torch_npu.contrib.DCNv2

> [!NOTICE]  
> This API is planned for deprecation. Use `torch_npu.contrib.ModulationDeformConv` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Applies an NPU-based modulated deformable 2D convolution operation. The implementation of `DCNv2` is designed and refactored based on MMCV.

## Prototype

```python
torch_npu.contrib.DCNv2(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True, pack=True)
```

## Parameters

- **`in_channels`** (`int`): Required. Number of channels in the input image.
- **`out_channels`** (`int`): Required. Number of output channels of the convolution.
- **`kernel_size`** (`int`, `tuple`): Required. Size of the convolution kernel.
- **`stride`** (`int`, `tuple`): Optional. Convolution stride. The default value is `1`.
- **`padding`** (`int` or `tuple`): Optional. Zero padding added to both sides of the input image. The default value is `0`.
- **`dilation`** (`int` or `tuple`): Optional. Spacing between convolution kernel elements. The default value is `1`.
- **`groups`** (`int`): Optional. Number of groups for the input and output channels. The default value is `1`.
- **`deformable_groups`** (`int`): Optional. Number of deformable groups. The default value is `1`.
- **`bias`** (`bool`): Optional. Specifies whether to add a bias to the output. Setting this parameter to `True` adds a bias to the output. The default value is `True`.
- **`pack`** (`bool`): Optional. Specifies whether to add `conv_offset` and `mask` to the model. Setting this parameter to `True` adds `conv_offset` and `mask` to the model. The default value is `True`.

## Constraints

`DCNv2` supports operations only under the `float32` data type. The weights and biases in `conv_offset` must be initialized to `0`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> x = torch.randn((2, 2, 5, 5), dtype=torch.float32).npu()
>>> x.requires_grad = True
>>> model = torch_npu.contrib.DCNv2(2, 2, 3, 2, 1).npu()
>>> output = model(x)
>>> output.sum().backward()
```
