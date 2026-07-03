# (beta) torch_npu.npu_convolution

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.nn.functional.conv2d` or `torch.nn.functional.conv3d` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Applies a 2D or 3D convolution to an input image composed of multiple input planes.

## Prototype

```python
torch_npu.npu_convolution(input, weight, bias, stride, padding, dilation, groups) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Input tensor with shape `(minibatch, in_channels, iH, iW)` or `(minibatch, in_channels, iT, iH, iW)`.
- **`weight`** (`Tensor`): Filter with shape `(out_channels, in_channels/groups, kH, kW)` or `(out_channels, in_channels/groups, kT, kH, kW)`.
- **`bias`** (`Tensor`): Bias with shape `(out_channels)`. This parameter can be `None`.
- **`stride`** (`List[int]`): Convolution kernel stride.
- **`padding`** (`List[int]`): Implicit padding on both sides of the input.
- **`dilation`** (`List[int]`): Kernel element spacing.
- **`groups`** (`int`): Groups the input. `in_channels` must be divisible by `groups`.
