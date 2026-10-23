# (beta) torch_npu.npu_deformable_conv2d

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

## Function

- Function: computes the output of the deformable convolution with the expected inputs.

- Formulas:

  Assume that the shape of the input (self) is [N, inC, inH, inW] and the shape of the output (out) is [N, outC, outH, outW]. outH and outW are computed based on the existing parameters:

  $$
  outH = (inH + padding[0] + padding[1] - ((K\_H - 1) * dilation[2] + 1)) // stride[2] + 1
  $$

  $$
  outW = (inW + padding[2] + padding[3] - ((K\_W - 1) * dilation[3] + 1)) // stride[3] + 1
  $$

  Sampling point subscripts of the standard convolution:

  $$
  x = -padding[2] + ow*stride[3] + kw*dilation[3], where the value of kw is (0, K\_W-1)
  $$

  $$
  y = -padding[0] + oh*stride[2] + kh*dilation[2], where the value of kh is (0, K\_H-1)
  $$

  Perform the deformable convolution based on the passed offset and compute the offset subscripts:

  $$
  (x,y) = (x + offsetX, y + offsetY)
  $$

  Use bilinear interpolation to compute the value of the offset point:

  $$
  (x_{0}, y_{0}) = (int(x), int(y)) \\
  (x_{1}, y_{1}) = (x_{0} + 1, y_{0} + 1)
  $$

  $$
  weight_{00} = (x_{1} - x) * (y_{1} - y) \\
  weight_{01} = (x_{1} - x) * (y - y_{0}) \\
  weight_{10} = (x - x_{0}) * (y_{1} - y) \\
  weight_{11} = (x - x_{0}) * (y - y_{0}) \\
  $$

  $$
  deformOut(x, y) = weight_{00} * self(x_{0}, y_{0}) + weight_{01} * self(x_{0}, y_{1}) + weight_{10} * self(x_{1}, y_{0}) + weight_{11} * self(x_{1}, y_{1})
  $$

  Perform the convolution computation to obtain the final output:

  $$
  \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{deformOut}(N_i, k)
  $$

## Prototype

```python
torch_npu.npu_deformable_conv2d(self, weight, offset, bias, kernel_size, stride, padding, dilation=[1,1,1,1], groups=1, deformable_groups=1, modulated=True) -> (Tensor, Tensor)
```

## Parameters

- **`self`** (`Tensor`): Required. Input image, $self$ in the formula. This parameter must be a 4D tensor. Empty tensors are not supported. Non-contiguous tensors are supported. The data layout can be NCHW or ND. The data is stored in the order of `[batch, in_channels, in_height, in_width]`. The product of `in_height * in_width` must be less than 2147483647.
- **`weight`** (`Tensor`): Required. Learnable filter, $weight$ in the formula. This parameter must be a 4D tensor. Empty tensors are not supported. Non-contiguous tensors are supported. The data layout can be NCHW or ND, and must be identical to that of `self`. The data is stored in the order of `[out_channels, in_channels/groups, filter_height, filter_width]`. `filter_height` indicates the height of the convolution kernel, and `filter_width` indicates the width of the convolution kernel.
- **`offset`** (`Tensor`): Required. X-y coordinate offsets and mask, $offset$ in the formula. This parameter must be a 4D tensor. Empty tensors are not supported. Non-contiguous tensors are supported. The data layout can be NCHW or ND, and must be identical to that of `self`. When `modulated` is `True`, the data is stored in the order of `[batch, deformable_groups * filter_height * filter_width * 3, out_height, out_width]`. When `modulated` is `False`, the data is stored in the order of `[batch, deformable_groups * filter_height * filter_width * 2, out_height, out_width]`.
- **`bias`** (`Tensor`): Optional. A 1D tensor containing the additive bias for the filter output, $bias$ in the formula. Empty tensors are not supported. Non-contiguous tensors are supported. The data layout is ND. `None` can be passed to indicate that no bias is used. If not `None`, the data is stored in the order of `[out_channels]`.
- **`kernel_size`** (`List[int]`): Required. Kernel size, $K_H$ and $K_W$ in the formula. This parameter must be a tuple or list of 2 integers `(K_H, K_W)`. Each element must be greater than 0. The product of `K_H * K_W` must not exceed 2048. The product of `K_H * K_W * in_channels/groups` must not exceed 65535.
- **`stride`** (`List[int]`): Required. Sliding window stride for each input dimension, $stride$ in the formula. This parameter must be a list of 4 integers. The dimension order is interpreted based on the data layout of `self`. Each element must be greater than 0. The N dimension and C dimension must be set to `1`.
- **`padding`** (`List[int]`): Required. Number of pixels added to each side (top, bottom, left, and right) of the input, $padding$ in the formula. This parameter must be a list of 4 integers.
- **`dilation`** (`List[int]`): Optional. Dilation factor for each input dimension, $dilation$ in the formula. This parameter must be a list of 4 integers. The dimension order is interpreted based on the data layout of `self`. Each element must be greater than 0. The N dimension and C dimension must be set to `1`. The default value is `[1, 1, 1, 1]`.
- **`groups`** (`int`): Optional. Number of groups connecting input channels to output channels. The data type must be `torch.int32`. Both `in_channels` and `out_channels` must be divisible by `groups`. The value of `groups` must be greater than 0. The default value is `1`.
- **`deformable_groups`** (`int`): Optional. Number of deformable group partitions. The data type must be `torch.int32`. `in_channels` must be divisible by `deformable_groups`. The value of `deformable_groups` must be greater than 0. The default value is `1`.
- **`modulated`** (`bool`): Optional. Specifies whether to include a mask in `offset`. Valid values are `True` (includes a mask) or `False` (excludes a mask). The default value is `True`.

## Return Values

- **`conv_output`** (`Tensor`): Result tensor after deformable convolution processing, $out$ in the formula. Empty tensors are not supported. Non-contiguous tensors are supported. The shape is `[batch, out_channels, out_height, out_width]`. The data layout must be identical to that of `self`.
- **`deformable_offset`** (`Tensor`): Offset tensor used to adjust sampling positions in deformable convolution, $deformOut$ in the formula. Empty tensors are not supported. Non-contiguous tensors are supported. The shape is `(batch, in_channels, out_height * K_H, out_width * K_W)`. The data layout must be identical to that of `self`.

## Constraints

- This API supports only single-operator mode calls.

<!-- npu="A3,910b,910,310p" id6 -->
- Atlas A3 training products, Atlas A2 training products, Atlas inference products, and Atlas training products: All Tensor type input parameters are automatically converted to `torch.float32` regardless of the input data type. The data type of the output parameter is `torch.float32`.
<!-- end id6 -->
<!-- npu="950" id7 -->
- Ascend 950PR&950DT products: The input and output data types are the same, and `torch.float16`, `torch.float32`, and `torch.bfloat16` are supported. The `modulated` parameter can only be set to True.
<!-- end id7 -->

## Example

```python
>>> import torch, torch_npu
>>> x = torch.rand(16, 32, 32, 32).npu()
>>> weight = torch.rand(32, 32, 5, 5).npu()
>>> offset = torch.rand(16, 75, 32, 32).npu()
>>> output, deform_offset = torch_npu.npu_deformable_conv2d(x, weight, offset, None, kernel_size=[5, 5], stride = [1, 1, 1, 1], padding = [2, 2, 2, 2])
>>> print(output.shape)
torch.Size([16, 32, 32, 32])
>>> print(deform_offset.shape)
torch.Size([16, 32, 160, 160])
```
