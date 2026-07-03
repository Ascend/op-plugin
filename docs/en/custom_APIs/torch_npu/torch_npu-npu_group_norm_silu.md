# torch_npu.npu_group_norm_silu

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas A2 inference products</term>   | √  |
|<term>Atlas inference products</term>   | √  |

## Function

- Description: Computes group normalization for the input tensor `input`. This API returns the output tensor `out`, the mean `meanOut`, the reciprocal of the standard deviation `rstdOut`, and the SiLU output.
- Formulas:
    - GroupNorm: $x$ indicates the input tensor `input`, and $\gamma$ and $\beta$ indicate `weight` and `bias`, respectively. $E[x] = \bar{x}$ indicates the mean of $x$, and $ Var[x]=\frac{1}{n}\sum_{i=1}^{n} (x_i - E[x])^2 $ indicates the variance of $x$.
    $$
    \begin{cases}
    \text{groupnormOut} = \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
    \text{meanOut}  = E[x] \\
    \text{rstdOut}  = \frac{1}{\sqrt{Var[x] + eps}}
    \end{cases}
    $$
    - Silu:
    $$
    \text{out} = \frac{\text{groupnormOut}}{1 + e^{-\text{groupnormOut}}}
    $$

## Prototype

```python
torch_npu.npu_group_norm_silu(input, weight, bias, group, eps=0.00001) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Source data tensor. This parameter must be 2D to 8D, and the size of its first dimension must be divisible by `group`. The data layout can be ND. Non-contiguous tensors are supported.
    - <term>Atlas inference products</term>: The data type can be `float16` or `float32`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float16`, `float32`, or `bfloat16`.

- **`weight`** (`Tensor`): Optional. Index tensor. This parameter must be a 1D tensor, and its element count must be identical to the size of the first dimension of `input`. The data layout can be ND. Non-contiguous tensors are supported.
    - <term>Atlas inference products</term>: The data type can be `float16` or `float32`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float16`, `float32`, or `bfloat16`.

- **`bias`** (`Tensor`): Optional. Tensor of updated data. This parameter must be a 1D tensor, and its element count must be identical to the size of the first dimension of `input`. The data layout can be ND. Non-contiguous tensors are supported.
    - <term>Atlas inference products</term>: The data type can be `float16` or `float32`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float16`, `float32`, or `bfloat16`.

- **`group`** (`int`): Required. Number of groups that the first dimension of `input` is divided into. The value of `group` must be greater than 0.
- **`eps`** (`float`): Optional. Value added to the denominator for numerical stability. To maintain precision, the value of `eps` must be greater than 0. The default value is `0.00001`.

## Return Values

- **`out`** (`Tensor`): Output tensor. The shape and data type are identical to those of `input`. The data layout can be ND. Non-contiguous tensors are supported.
    - <term>Atlas inference products</term>: The data type can be `float16` or `float32`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float16`, `float32`, or `bfloat16`.

- **`meanOut`** (`Tensor`): The data type is identical to that of `input`. The shape of this parameter is `(N, group)`, where $N$ indicates the size of the 0th dimension of `input`. The data layout can be ND. Non-contiguous tensors are supported.
    - <term>Atlas inference products</term>: The data type can be `float16` or `float32`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float16`, `float32`, or `bfloat16`.

- **`rstdOut`** (`Tensor`): The data type is identical to that of `input`. The shape of this parameter is `(N, group)`, where $N$ indicates the size of the 0th dimension of `input`. The data layout can be ND. Non-contiguous tensors are supported.
    - <term>Atlas inference products</term>: The data type can be `float16` or `float32`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float16`, `float32`, or `bfloat16`.

## Constraints

- This API can be used in inference and training scenarios.
- The data types of `input`, `weight`, `bias`, `out`, `meanOut`, and `rstdOut` must be within the supported ranges.
- The data types of `out`, `meanOut`, and `rstdOut` must be identical to that of `input`. The data types of `weight` and `bias` can differ from that of `input`.
- The data types of `weight` and `bias` must be identical, and their precision must not be lower than that of `input`.
- `weight` and `bias` are 1D, and the element count must be identical to the size of the first dimension of `input`.
- `input` must be 2D to 8D, and the size of its first dimension must be divisible by `group`.
- The size of any dimension of `input` must be greater than 0.
- The shape of `out` must be identical to that of `input`.
- The shapes of `meanOut` and `rstdOut` must be `(N, group)`, where $N$ indicates the size of the 0th dimension of `input`.
- `eps` must be greater than 0.
- `group` must be greater than 0.

## Examples

```python
import torch
import numpy as np
import torch_npu
     
dtype = np.float32
shape_x = [24,320,48,48]
num_groups = 32
shape_c = [320]
eps = 0.00001
     
input_npu=torch.randn(shape_x,dtype=torch.float32).npu()
weight_npu=torch.randn(shape_c,dtype=torch.float32).npu()
bias_npu=torch.randn(shape_c,dtype=torch.float32).npu()
out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(input_npu, weight_npu, bias_npu, group=num_groups, eps=eps)
     
     
input_npu=torch.randn(shape_x,dtype=torch.bfloat16).npu()
weight_npu=torch.randn(shape_c,dtype=torch.bfloat16).npu()
bias_npu=torch.randn(shape_c,dtype=torch.bfloat16).npu()
out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(input_npu, weight_npu, bias_npu, group=num_groups, eps=eps)
     
input_npu=torch.randn(shape_x,dtype=torch.float16).npu()
weight_npu=torch.randn(shape_c,dtype=torch.float16).npu()
bias_npu=torch.randn(shape_c,dtype=torch.float16).npu()
out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(input_npu, weight_npu, bias_npu, group=num_groups, eps=eps)
```
