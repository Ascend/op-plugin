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

- **`input`** (`Tensor`): Required. Input tensor with shape `(minibatch, in_channels, iH, iW)`.
- **`weight`** (`Tensor`): Required. Filter tensor with shape `(in_channels, out_channels/groups, kH, kW)`.
- **`bias`** (`Tensor`): Optional. Bias tensor with shape `(out_channels)`. The default value is `None`.
- **`padding`** (`List[int]`): Optional. Zero padding applied to both sides of each dimension of the input shape, with the number of padding elements given by `(dilation * (kernel_size - 1) - padding)`. The default value is `[0, 0]`.
- **`output_padding`** (`List[int]`): Optional. Additional size added to one side of each dimension of the output shape. The default value is `[0, 0]`.
- **`stride`** (`List[int]`): Optional. Convolution kernel stride. The default value is `[1, 1]`.
- **`dilation`** (`List[int]`): Optional. Spacing between kernel elements. The default value is `[1, 1]`.
- **`groups`** (`int`): Optional. Number of groups into which the input is divided. `in_channels` and `out_channels` must both be divisible by `groups`. The default value is `1`.
