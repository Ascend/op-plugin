# (beta) torch_npu.npu_deformable_conv2d

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

- Description: Computes the output of deformable convolution using the expected input.

- Formulas:
  
  Assume that the shape of the input (`self`) is `[N, inC, inH, inW]` and that of the output (`out`) is `[N, outC, outH, outW]`. Output height ($outH$) and width ($outW$) are calculated based on the existing parameters:
  
  $$
  outH = (inH + padding[0] + padding[1] - ((K_H - 1) * dilation[2] + 1)) // stride[2] + 1
  $$
  
  $$
  outW = (inW + padding[2] + padding[3] - ((K_W - 1) * dilation[3] + 1)) // stride[3] + 1
  $$
  
  Sampling point index computation for standard convolution:
  
  $$
  x = -padding[2] + ow*stride[3] + kw*dilation[3], kw ∈ (0, K\_W – 1)
  $$
  
  $$
  y = -padding[0] + oh*stride[2] + kh*dilation[2], kh ∈ (0, K\_H – 1)
  $$
  
  Offset index calculation after deformable convolution based on the provided offset:
  
  $$
  (x,y) = (x + offsetX, y + offsetY)
  $$

  Bilinear interpolation calculation for the value at the deformed location:
  
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
  deformOut(x, y) = weight_{00} * self(x0, y0) + weight_{01} * self(x0,y1) + weight_{10} * self(x1, y0) + weight_{11} * self(x1,y1)
  $$
  
  Final output computation through convolution:
  
  $$
  \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{deformOut}(N_i, k)
  $$

## Prototype

```python
torch_npu.npu_deformable_conv2d(self, weight, offset, bias=None, kernel_size, stride, padding, dilation=[1,1,1,1], groups=1, deformable_groups=1, modulated=True) -> (Tensor, Tensor)
```

## Parameters

- **`self`** (`Tensor`): Required. Input image, $self$ in the formula. This parameter must be a 4D tensor. Empty tensors are not supported. Non-contiguous tensors are supported. The data layout can be NCHW or ND. The data is stored in the order of `[batch, in_channels, in_height, in_width]`. The product of `in_height * in_width` must be less than 2147483647.
- **`weight`** (`Tensor`): Required. Learnable filter, $weight$ in the formula. This parameter must be a 4D tensor. Empty tensors are not supported. Non-contiguous tensors are supported. The data layout can be NCHW or ND, and must be identical to that of `self`. The data is stored in the order of `[out_channels, in_channels/groups, filter_height, filter_width]`. `filter_height` indicates the height of the convolution kernel, and `filter_width` indicates the width of the convolution kernel.
- **`offset`** (`Tensor`): Required. X-y coordinate offsets and mask, $offset$ in the formula. This parameter must be a 4D tensor. Empty tensors are not supported. Non-contiguous tensors are supported. The data layout can be NCHW or ND, and must be identical to that of `self`. When `modulated` is `True`, the data is stored in the order of `[batch, deformable_groups * filter_height * filter_width * 3, out_height, out_width]`. When `modulated` is `False`, the data is stored in the order of `[batch, deformable_groups * filter_height * filter_width * 2, out_height, out_width]`.
- **`bias`** (`Tensor`): Optional. Additive bias for the filter output, $bias$ in the formula. This parameter must be a 1D tensor. Empty tensors are not supported. Non-contiguous tensors are supported. The data layout is ND. The default value is `None`. If set to non-`None`, the data is stored in the order of `[out_channels]`.
- **`kernel_size`** (`List[int]`): Required. Kernel size, $K_H$ and $K_W$ in the formula. This parameter must be a tuple or list of 2 integers `(K_H, K_W)`. Each element must be greater than 0. The product of `K_H * K_W` must not exceed 2048. The product of `K_H * K_W * in_channels/groups` must not exceed 65535.
- **`stride`** (`List[int]`): Required. Sliding window stride for each input dimension, $stride$ in the formula. This parameter must be a list of 4 integers. The dimension order is interpreted based on the data layout of `self`. Each element must be greater than 0. The N dimension and C dimension must be set to `1`.
- **`padding`** (`List[int]`): Required. Number of pixels added to each side (top, bottom, left, and right) of the input, $padding$ in the formula. This parameter must be a list of 4 integers.
- **`dilation`** (`List[int]`): Optional. Dilation factor for each input dimension, $dilation$ in the formula. This parameter must be a list of 4 integers. The dimension order is interpreted based on the data layout of `self`. Each element must be greater than 0. The N dimension and C dimension must be set to `1`. The default value is `[1, 1, 1, 1]`.
- **`groups`** (`int`): Optional. Number of groups connecting input channels to output channels. The data type must be `int32`. Both `in_channels` and `out_channels` must be divisible by `groups`. The value of `groups` must be greater than 0. The default value is `1`.
- **`deformable_groups`** (`int`): Optional. Number of deformable group partitions. The data type must be `int32`. `in_channels` must be divisible by `deformable_groups`. The value of `deformable_groups` must be greater than 0. The default value is `1`.
- **`modulated`** (`bool`): Optional. Specifies whether to include a mask in `offset`. Valid values are `True` (includes a mask) or `False` (excludes a mask). The default value is `True`.

## Return Values

- **`conv_output`** (`Tensor`): Result tensor after deformable convolution processing, $out$ in the formula. Empty tensors are not supported. Non-contiguous tensors are supported. The shape is `[batch, out_channels, out_height, out_width]`. The data layout must be identical to that of `self`.
- **`deformable_offset`** (`Tensor`): Offset tensor used to adjust sampling positions in deformable convolution, $deformOut$ in the formula. Empty tensors are not supported. Non-contiguous tensors are supported. The shape is `(batch, in_channels, out_height * K_H, out_width * K_W)`. The data layout must be identical to that of `self`.

## Constraints

All tensor inputs are automatically converted to `float32` regardless of their original data types. The output data type is `float32`.

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
