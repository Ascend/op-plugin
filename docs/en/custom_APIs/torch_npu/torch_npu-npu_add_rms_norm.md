# torch_npu.npu_add_rms_norm

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>   |     √    |

## Function

- Description: Fuses Add computation with RMSNorm normalization, commonly used in foundation models to normalize tensors after residual connections.
- Formulas:

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
  \operatorname{RMSNorm}(x_i)=\frac{x_i}{\operatorname{RMS}(\mathbf{x})} gamma_i, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

## Prototype

```python
torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=1e-06) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`x1`** (`Tensor`): Required. First input for Add computation, $x1$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type can be `float32`, `float16`, or `bfloat16`. The shape must have 1 to 8 dimensions.
- **`x2`** (`Tensor`): Required. Second input for Add computation, $x2$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type can be `float32`, `float16`, or `bfloat16`. The shape must have 1 to 8 dimensions.
- **`gamma`** (`Tensor`): Required. Scaling factor (weight) for RMSNorm, $gamma$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type must be identical to that of `x1`. The shape must be identical to the trailing dimensions of `x1`, which correspond to the dimensions to be normalized.
- **`epsilon`** (`float`): Optional. Value added to the denominator to ensure numerical stability, $epsilon$ in the formulas. The default value is `1e-6`.

## Return Values

- **`yOut`** (`Tensor`): Final output, $RMSNorm(x)$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type and shape must be identical to those of the input `x1`.

- **`rstdOut`** (`Tensor`): Reciprocal of the normalized standard deviation, reciprocal of $RMS(x)$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type can be `float32`. The shape must be identical to the leading dimensions of `x1`. The leading dimensions refer to the dimensions that do not require normalization. Examples of the relationship among the shapes of `x1`, `gamma`, and `rstdOut` are as follows:
  - If `x1` has shape `(2, 3, 4, 8)` and `gamma` has shape `(8,)`, `rstdOut` has shape `(2, 3, 4, 1)`.
  - If `x1` has shape `(2, 3, 4, 8)` and `gamma` has shape `(4, 8)`, `rstdOut` has shape `(2, 3, 1, 1)`.

- **`xOut`** (`Tensor`): Result of Add computation, $x$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type and shape must be identical to those of the input `x1`.

## Constraints

- Boundary value scenarios:
  - When the input is `Inf`, the output is `Inf`.
  - When the input is `NaN`, the output is `NaN`.
- Atlas inference products:
  - The parameters `x1`, `x2`, `gamma`, `yOut`, and `xOut` do not support the `bfloat16` data type.
  - The parameter `rstdOut` is invalid in current product usage scenarios.

## Example

```python
import torch
import torch_npu

x1 = torch.rand(4, 8, dtype=torch.float16, device='npu') * 100
x2 = torch.rand(4, 8, dtype=torch.float16, device='npu') * 100
gamma = torch.rand(8, dtype=torch.float16, device='npu') * 100
epsilon = 1e-6

y, rstd, x = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=epsilon)

print("y:", y)
print("y.shape:", y.shape)
print("y.dtype:", y.dtype)
print("rstd:", rstd)
print("rstd.dtype:", rstd.dtype)
print("x:", x)
```
