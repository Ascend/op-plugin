# (beta) torch_npu.npu_convolution_transpose

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.nn.functional.conv_transpose2d` or `torch.nn.functional.conv_transpose3d` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Applies a 2D or 3D transposed convolution operator to an input image composed of multiple input planes. Sometimes, this process is also referred to as deconvolution.

## Prototype

```python
torch_npu.npu_convolution_transpose(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Input tensor with shape `(minibatch, in_channels, iH, iW)` or `(minibatch, in_channels, iT, iH, iW)`.
- **`weight`** (`Tensor`): Filter tensor with shape `(in_channels, out_channels/groups, kH, kW)` or `(in_channels, out_channels/groups, kT, kH, kW)`.
- **`bias`** (`Tensor`): Optional. Bias tensor with shape `(out_channels)`.
- **`padding`** (`List[int]`): Zero padding applied to both sides of each dimension of the input, based on `(dilation * (kernel_size - 1) - padding)`.
- **`output_padding`** (`List[int]`): Additional size added to one side of each dimension of the output shape.
- **`stride`** (`List[int]`): Convolution kernel stride.
- **`dilation`** (`List[int]`): Spacing between kernel elements.
- **`groups`** (`int`): Number of groups into which the input is divided. `in_channels` must be divisible by `groups`.
