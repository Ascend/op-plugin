# torch_npu.npu_group_norm_swish

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |

## Function

- Description: Performs fused computation of group normalization (GroupNorm) and Swish activation for the input tensor `input`. This API generates the group normalization result `y`, mean `mean`, reciprocal standard deviation `rstd`, and the Swish-activated output.
- Formulas:
  - GroupNorm: $x$ indicates the input tensor `input`, $E[x] = \bar{x}$ indicates the mean of $x$, and $Var[x] = \frac{1}{n} * \sum_{i=1}^{n} (x_i - E[x])^2$ indicates the variance of $x$. $\gamma$ indicates `weight` and $\beta$ indicates `bias`.
  $$
  \begin{cases}
  y & = \frac{x - E[x]}{\sqrt{{Var[x]} + eps}} * \gamma + \beta \\ 
  mean & = E[x] \\ 
  rstd & = \frac{1}{\sqrt{{Var[x]} + eps}}
  \end{cases}
  $$

  - Swish: The input $x$ in the Swish formula is the output $y$ obtained from the GroupNorm formula.
  $$
  y = \frac{x}{1 + e^{-scale \cdot x}}
  $$
  
> **Note**:<br>
> When computing backward gradients, if deterministic outputs are required to eliminate randomness, enable deterministic computation. For details, see [Deterministic Computation Switch](../determin_API_list.md).

## Prototype

```python
torch_npu.npu_group_norm_swish(input, num_groups, weight, bias, eps=1e-5, swish_scale=1.0) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Data to be normalized by group. This parameter is 2D to 8D. The data type can be `float16`, `float32`, or `bfloat16`.

- **`num_groups`** (`int`): Required. Number of groups that the first dimension of `input` is divided into. The size of the first dimension of `input` must be divisible by `num_groups`.
  - `num_groups=1`: equivalent to LayerNorm (layer normalization), which normalizes the entire input. This configuration is applicable to scenarios such as sequence modeling and after fully connected layers.
  - `num_groups=C` (where `C` is the number of channels): equivalent to InstanceNorm (instance normalization), which normalizes each channel independently. This configuration is applicable to scenarios such as style transfer and image generation.
  - `1 < num_groups < C`: standard GroupNorm, which divides channels into multiple groups along the channel dimension for normalization. The default configuration is `num_groups=32`, which is commonly used.
  > [!NOTE]
  > 
  > When backward gradients are computed, the result of $input.shape[1]/num_groups$ must not exceed 4000. Violating this constraint can cause errors during training.

- **`weight`** (`Tensor`): Required. Weight tensor. This parameter must be a 1D tensor, and the size of its 0th dimension must be identical to that of the first dimension of `input`. The data type can be `float16`, `float32`, or `bfloat16`, which must be identical to that of `input`.

- **`bias`** (`Tensor`): Required. Bias tensor. This parameter must be a 1D tensor, and the size of its 0th dimension must be identical to that of the first dimension of `input`. The data type can be `float16`, `float32`, or `bfloat16`, which must be identical to that of `input`.

- **`eps`** (`float`): Optional. Value added to the denominator for numerical stability during group normalization computation. The default value is `1e-5`.

- **`swish_scale`** (`float`): Optional. Scaling factor for Swish computation. The default value is `1.0`.

## Return Values

- **`y`** (`Tensor`): Final output after group normalization and Swish activation, which is used for network forward propagation. The shape and data type are identical to those of `input`. The data type can be `float16`, `float32`, or `bfloat16`.

- **`mean`** (`Tensor`): Mean value of each group, which is used for gradient computation during backward propagation and must be saved together with `y`. The shape of this parameter is `(N, num_groups)`, where $N$ indicates the size of the 0th dimension of `input`. The data type is identical to that of `input`.

- **`rstd`** (`Tensor`): Reciprocal standard deviation of each group, which is used for gradient calculation during backward propagation and must be saved together with `y`. The shape of this parameter is `(N, num_groups)`, where $N$ indicates the size of the 0th dimension of `input`. The data type is identical to that of `input`.

## Constraints

- When **backward gradients** are computed, the result of $input.shape[1]/num_groups$ must not exceed 4000. Violating this constraint can cause errors during training. This constraint takes effect only in forward and backward propagation scenarios. Inference-only scenarios are not limited by this constraint.
- The `input`, `weight`, and `bias` parameters must not contain `-inf`, `inf`, or `nan` values.

## Examples

```python
import torch
import torch_npu
 
input = torch.randn(3, 4, 6, dtype=torch.float32).npu()
weight = torch.randn(input.size(1), dtype=torch.float32).npu()
bias = torch.randn(input.size(1), dtype=torch.float32).npu()
num_groups = input.size(1)
eps = 1e-5
swish_scale = 1.0
out, mean, rstd = torch_npu.npu_group_norm_swish(input, num_groups, weight, bias, eps=eps, swish_scale=swish_scale)
```
