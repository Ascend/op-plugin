# torch_npu.npu_add_rms_norm_dynamic_quant

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |

## Function

- Description: The RMSNorm operator is a normalization operation commonly used in foundation models. Compared with the LayerNorm operator, it removes the mean subtraction step. The DynamicQuant operator performs symmetric dynamic quantization on the input tensor. The AddRmsNormDynamicQuant operator fuses the Add operator before RMSNorm and 1 or 2 DynamicQuant operators applied to the RMSNorm normalization output, reducing data transfer operations.

- Formulas:

  $$
  x=x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RMSNorm}(x)=\frac{x}{\operatorname{RMS}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  input1 =\begin{cases}
    y\cdot smoothScale1Optional & \ \ smoothScale1Optional \\
    y & !\ smoothScale1Optional
    \end{cases}
  $$

  $$
  input2 =\begin{cases}
    y\cdot smoothScale2Optional & \ \ smoothScale2Optional \\
    y & !\ smoothScale2Optional
    \end{cases}
  $$

  $$
  scale1Out=\begin{cases}
    row\_max(abs(input1))/127 & outputMask[0]=True\ ||\ !outputMask \\
    Invalid output & outputMask[0]=False
    \end{cases}
  $$

  $$
  y1Out=\begin{cases}
    round(input1/scale1Out) & outputMask[0]=True\ ||\ !outputMask \\
    Invalid output & outputMask[0]=False
    \end{cases}
  $$

$$
  scale2Out=\begin{cases}
    row\_max(abs(input2))/127 & outputMask[1]=True\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ smoothScale2Optional) \\
    Invalid output & outputMask[1]=False\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ !smoothScale2Optional)
    \end{cases}
$$

$$
  y2Out=\begin{cases}
    round(input2/scale2Out) & outputMask[1]=True\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ smoothScale2Optional)\\
    Invalid output & outputMask[1]=False\ ||\ (!outputMask\ \&\ smoothScale1Optional\ \&\ !smoothScale2Optional)
    \end{cases}
$$

  In the formulas, `row_max` represents the maximum value of each row.

## Prototype

```python
torch_npu.npu_add_rms_norm_dynamic_quant(x1, x2, gamma, *, smooth_scale1=None, smooth_scale2=None, beta=None, epsilon=1e-6, output_mask=[], y_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`x1`** (`Tensor`): Required. First input for Add computation, $x1$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type can be `float16` or `bfloat16`. The shape must have 2 to 8 dimensions.
- **`x2`** (`Tensor`): Required. Second input for Add computation, $x2$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type can be `float16` or `bfloat16`. The shape must be identical to that of `x1`.
- **`gamma`** (`Tensor`): Required. Scaling factor (weight) for RMSNorm, $gamma$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type must be identical to that of `x1`. The shape must be 1-dimensional, and the number of elements must equal the size of the last dimension of `x1`.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`smooth_scale1`** (`Tensor`): Optional. Smooth scaling factor for the first quantization path, $smoothScale1$ in the formulas. The data type must be identical to that of `x1`. The shape must be 1-dimensional, and the number of elements must equal the size of the last dimension of `x1`. The default value is `None`. If set to `None`, no smooth operation is executed on the quantization branch.
- **`smooth_scale2`** (`Tensor`): Optional. Smooth scaling factor for the second quantization path, $smoothScale2$ in the formulas. The data type must be identical to that of `x1`. The shape must be 1-dimensional, and the number of elements must equal the size of the last dimension of `x1`. The default value is `None`. If set to `None`, no smooth operation is executed on the quantization branch.
- `beta` (`Tensor`): Optional. Bias term of RMSNorm, $beta$ in the formulas. The data type must be identical to that of `x1`. The shape must be 1-dimensional, and the number of elements must equal the size of the last dimension of `x1`. The default value is `None`. If set to `None`, no bias is added.
- **`epsilon`** (`float`): Optional. Value added to the denominator to ensure numerical stability, $epsilon$ in the formulas. The default value is `1e-6`.
- **`output_mask`** (`bool[2]`): Optional. Boolean array of length 2 used to control whether to compute the two quantization outputs, $outputMask$ in the formulas. `output_mask[0]` controls the first quantization output (`y1`, `scale1`), and `output_mask[1]` controls the second quantization output (`y2`, `scale2`).
- **`y_dtype`** (`ScalarType`): Optional. Quantized output data type of `y1` and `y2`. `None` or `torch.int8` indicates `int8`. `torch.quint4x2` indicates `int4`. For the `int4` scenario, the last dimension of `x1` must be divisible by 8. The default value is `None`.

## Return Values

- **`y1`** (`Tensor`): Output tensor of the first dynamic quantization path, $y1Out$ in the formulas. When `output_mask[0]` is `True`, the data type can be `int8` or `int4`, and the shape must be identical to that of `x1`. When `output_mask[0]` is `False`, an empty tensor is returned.
- **`y2`** (`Tensor`): Output tensor of the second dynamic quantization path, $y2Out$ in the formulas. When `output_mask[1]` is `True`, the data type can be `int8` or `int4`, and the shape must be identical to that of `x1`. When `output_mask[1]` is `False`, an empty tensor is returned.
- **`x_out`** (`Tensor`): Result of Add computation, $x$ in the formulas. The data type and shape must be identical to those of the input `x1`.
- **`scale1`** (`Tensor`): Scaling factor of the first dynamic quantization path, $scale1Out$ in the formulas. When `output_mask[0]` is `True`, the data type is `float32`, and the shape is the shape of `x1` with the last dimension removed. When `output_mask[0]` is `False`, an empty tensor is returned.
- **`scale2`** (`Tensor`): Scaling factor of the second dynamic quantization path, $scale2Out$ in the formulas. When `output_mask[1]` is `True`, the data type is `float32`, and the shape is the shape of `x1` with the last dimension removed. When `output_mask[1]` is `False`, an empty tensor is returned.

## Constraints

- When `output_mask` is not empty: if `smooth_scale1` is provided, `output_mask[0]` must be `True`. If `smooth_scale2` is provided, `output_mask[1]` must be `True`.
- When `output_mask` is not empty, `output_mask[0]` and `output_mask[1]` cannot both be `False`.

## Example

```python
import torch
import torch_npu

x1 = torch.randn(2, 3, 32, dtype=torch.float16, device='npu')
x2 = torch.randn(2, 3, 32, dtype=torch.float16, device='npu')
gamma = torch.ones(32, dtype=torch.float16, device='npu')
beta = torch.zeros(32, dtype=torch.float16, device='npu')
smooth_scale1 = torch.ones(32, dtype=torch.float16, device='npu')
smooth_scale2 = torch.ones(32, dtype=torch.float16, device='npu')
epsilon = 1e-6

y1, y2, x_out, scale1, scale2 = torch_npu.npu_add_rms_norm_dynamic_quant(
    x1, x2, gamma,
    smooth_scale1=smooth_scale1,
    smooth_scale2=smooth_scale2,
    beta=beta,
    epsilon=epsilon,
    output_mask=[True, True],
)

print("y1:", y1)
print("y1.shape:", y1.shape)
print("y1.dtype:", y1.dtype)
print("y2:", y2)
print("y2.shape:", y2.shape)
print("y2.dtype:", y2.dtype)
print("x_out:", x_out)
print("x_out.shape:", x_out.shape)
print("x_out.dtype:", x_out.dtype)
print("scale1:", scale1)
print("scale1.shape:", scale1.shape)
print("scale2:", scale2)
print("scale2.shape:", scale2.shape)
```
