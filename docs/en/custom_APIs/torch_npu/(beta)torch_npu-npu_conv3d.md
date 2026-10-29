# (beta) torch_npu.npu_conv3d

> [!NOTICE]  
> This API is planned for deprecation. Use the `torch.nn.functional.conv3d` API instead.

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->

## Function

Applies a 3D convolution to an input image composed of multiple input planes.

## Prototype

```python
torch_npu.npu_conv3d(input, weight, bias, stride, padding, dilation, groups) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Input tensor with shape `(minibatch, in_channels, iT, iH, iW)`.
- **`weight`** (`Tensor`): Required. Filter tensor with shape `(out_channels, in_channels/groups, kT, kH, kW)`.
- **`bias`** (`Tensor`): Optional. Bias tensor with shape `(out_channels)`. The default value is `None`.
- **`stride`** (`List[int]`): Optional. Convolution kernel stride. The default value is `[1, 1, 1]`.
- **`padding`** (`List[int]`): Optional. Implicit padding on both sides of the input. The default value is `[1, 1, 1]`.
- **`dilation`** (`List[int]`): Optional. Spacing between kernel elements. The default value is `[1, 1, 1]`.
- **`groups`** (`int`): Optional. Number of groups into which the input is divided. `in_channels` must be divisible by `groups`. The default value is `1`.
