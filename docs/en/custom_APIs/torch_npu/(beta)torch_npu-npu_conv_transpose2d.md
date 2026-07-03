# (beta) torch_npu.npu_conv_transpose2d

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.nn.functional.conv_transpose2d` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Applies a 2D transposed convolution operator to an input image composed of multiple input planes. Sometimes, this process is also referred to as deconvolution.

## Prototype

```python
torch_npu.npu_conv_transpose2d(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Input tensor with shape `(minibatch, in_channels, iH, iW)`.
- **`weight`** (`Tensor`): Filter with shape `(in_channels, out_channels/groups, kH, kW)`.
- **`bias`** (`Tensor`): Optional. Bias tensor with shape `(out_channels)`.
- **`padding`** (`List[int]`): Pads the edges of the input matrix with zeros based on `(dilation * (kernel_size - 1) - padding)`.
- **`output_padding`** (`List[int]`): Additional size added to one side of each dimension of the output shape.
- **`stride`** (`List[int]`): Convolution kernel stride.
- **`dilation`** (`List[int]`): Kernel element spacing.
- **`groups`** (`int`): Groups the input. `in_channels` must be divisible by `groups`.
