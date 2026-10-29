# torch_npu.npu_group_norm_silu

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->

## Function

- **API function**: Performs GroupNorm and SiLU activation on the input Tensor `input` in sequence and returns three Tensors: out (output after SiLU activation), meanOut (normalization mean), and rstdOut (reciprocal of the normalization standard deviation).
- **Formulas**:
    - GroupNorm: $x$ is the input `input`. In the formula, $γ$ and $β$ correspond to the optional parameters `weight` and `bias` respectively. When both `weight` and `bias` are empty, no affine transformation is performed (equivalent to $γ = 1$ and $β = 0$). $E[x] = \bar{x}$ represents the mean of $x$, and $Var[x] = \frac{1}{n}\sum_{i=1}^{n} (x_i - E[x])^2$ represents the variance of $x$. Then,
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

    Here, $n$ indicates the number of elements involved in the normalization in each group of a single sample. The mean and variance are calculated for each group.

## Prototype

```python
torch_npu.npu_group_norm_silu(input, weight, bias, group, eps=0.00001) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Source data Tensor. The number of dimensions must be 2 to 8, and the first dimension must be divisible by `group`. The data layout can be ND. Non-contiguous Tensors are supported.

    <!-- npu="310p" id4 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`, `torch.float32`.
    <!-- end id4 -->
    <!-- npu="910b" id5 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`.
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`.
    <!-- end id6 -->

- **`weight`** (`Tensor`): Optional. Scaling Tensor. The number of dimensions must be 1, and the number of elements must be the same as that of the first dimension of the input `input`. The data layout can be ND. Non-contiguous Tensors are supported.

    <!-- npu="310p" id7 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`, `torch.float32`.
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`.
    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`.
    <!-- end id9 -->

- **`bias`** (`Tensor`): Optional. Offset Tensor. The number of dimensions must be 1, and the number of elements must be the same as that of the first dimension of the input `input`. The data layout can be ND. Non-contiguous Tensors are supported.

    <!-- npu="310p" id10 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`, `torch.float32`.
    <!-- end id10 -->
    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`.
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`.
    <!-- end id12 -->

- **`group`** (`int`): Required. Divides the first dimension of the input `input` into `group` groups. The value of `group` must be greater than 0. The data type can be `int64`.

- **`eps`** (`float`): Optional. Value added to the denominator to maintain numerical stability. To maintain precision, the value of `eps` must be greater than 0. The default value is 0.00001. The data type can be `float32`.

## Return Values

- **`out`** (`Tensor`): The data type and shape are the same as those of `input`. ND is supported. Non-contiguous Tensors are supported.

    <!-- npu="310p" id13 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`, `torch.float32`.
    <!-- end id13 -->
    <!-- npu="910b" id14 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`.
    <!-- end id14 -->
    <!-- npu="950" id15 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`.
    <!-- end id15 -->

- **`meanOut`** (`Tensor`): The data type is the same as that of `input`, and the shape is `(N, group)`, where N is the value of dimension 0 of `input`. The data layout can be ND. Non-contiguous Tensors are supported.

    <!-- npu="310p" id16 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`, `torch.float32`.
    <!-- end id16 -->
    <!-- npu="910b" id17 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`.
    <!-- end id17 -->
    <!-- npu="950" id18 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`.
    <!-- end id18 -->

- **`rstdOut`** (`Tensor`): The data type is the same as that of `input`, and the shape is `(N, group)`, where N is the value of dimension 0 of `input`. The data layout can be ND. Non-contiguous Tensors are supported.

    <!-- npu="310p" id19 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`, `torch.float32`.
    <!-- end id19 -->
    <!-- npu="910b" id20 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`.
    <!-- end id20 -->
    <!-- npu="950" id21 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`.
    <!-- end id21 -->

## Constraints

- This API supports inference and training scenarios.
- The data types of `input`, `weight`, `bias`, `out`, `meanOut`, and `rstdOut` must be within the supported range.
- The data types of `out`, `meanOut`, and `rstdOut` are the same as that of `input`; the data types of `weight` and `bias` can be different from that of `input`.
- The data types of `weight` and `bias` must be the same, and their precision must not be lower than that of the data type of `input`.
- The number of dimensions of `weight` and `bias` must be 1, and the number of elements must be the same as that of the first dimension of the input `input`.
- The number of dimensions of `input` must be greater than 1 and less than or equal to 8, and the first dimension of `input` must be divisible by `group`.
- Every dimension of `input` must be greater than 0.
- The shape of `out` is the same as that of `input`.
- The shape of `meanOut` and `rstdOut` is `(N, group)`, where N is the value of dimension 0 of `input`.
- The value of `eps` must be greater than 0.
- The value of `group` must be greater than 0.

## Examples

- Single-operator call

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

<!-- npu="950" id22 -->
- Graph mode call: Applicable only to <term>Ascend 950PR&950DT products</term>.

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, weight, bias, group, eps):
            return torch_npu.npu_group_norm_silu(input, weight, bias, group, eps)

    x = torch.randn(10, 1024, 4, 8, dtype=torch.float32).npu()
    weight = torch.randn(1024, dtype=torch.float32).npu()
    bias = torch.randn(1024, dtype=torch.float32).npu()
    model = Net().npu()
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    out, mean, rstd = model(x, weight, bias, group=4, eps=0.0001)
    ```
<!-- end id22 -->
