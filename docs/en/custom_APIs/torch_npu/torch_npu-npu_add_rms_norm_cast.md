# torch_npu.npu_add_rms_norm_cast

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- **API function**: RmsNorm is a normalization operation commonly used in foundation models. AddRmsNormCast fuses the Cast operator after AddRmsNorm to reduce memory load and store operations.

- **Formulas**:

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
  y1=\operatorname{RMSNorm}(x_i)=\frac{x_i}{\operatorname{RMS}(\mathbf{x})} gamma_i, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  y2=cast(y1)
  $$

## Prototype

```python
torch_npu.npu_add_rms_norm_cast(x1, x2, gamma, epsilon=1e-06) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`x1`** (`Tensor`): Required. Original data input to be normalized, $x1$ in the formulas. The data layout can be ND. Empty and non-contiguous Tensors are supported. The data type can be `torch.float16` or `torch.bfloat16`. The shape can have 1 to 8 dimensions.
- **`x2`** (`Tensor`): Required. Original data input to be normalized, $x2$ in the formulas. The data layout can be ND. Empty and non-contiguous Tensors are supported. The shape and data type must be the same as those of `x1`.
- **`gamma`** (`Tensor`): Required. Weight Tensor in the normalization process, $gamma$ in the formulas. The data layout can be ND. Empty and non-contiguous Tensors are supported. The data type must be the same as that of `x1`. The shape can have 1 to 8 dimensions and must be the same as the dimensions of `x1` that require normalization.
- **`epsilon`** (`float`): Optional. Input $epsilon$ in the formulas, used to prevent division by zero. The data type is `double`. A small positive value is recommended. The default value is `1e-6`.

## Return Values

- **`y1`** (`Tensor`): Output Tensor after normalization and type conversion, $y1$ in the formulas. The data layout can be ND. Empty and non-contiguous Tensors are supported. The data type can be `torch.float32`, and the shape is the same as that of the input `x1`.
- **`y2`** (`Tensor`): Output Tensor after normalization, $y2$ in the formulas. The data layout can be ND. Empty and non-contiguous Tensors are supported. The shape and data type must be the same as those of the input `x1`.
- **`rstd`** (`Tensor`): Standard deviation of `x`, corresponding to $RMS(x)$ in the formulas. The data layout can be ND. Empty and non-contiguous Tensors are supported. The data type can be `torch.float32`. The number of dimensions must be the same as that of `x1`. The dimensions that do not require normalization have the same sizes as the corresponding dimensions of `x1`, and the dimensions that require normalization have a size of 1. When the input `x1` is an empty Tensor, `rstd` must also be an empty Tensor. Examples of the relationship among the shapes of `x1`, `gamma`, and `rstd`:
  - If `x1` has shape `(2, 3, 4, 8)` and `gamma` has shape `(8,)`, `rstd` has shape `(2, 3, 4, 1)`.
  - If `x1` has shape `(2, 3, 4, 8)` and `gamma` has shape `(4, 8)`, `rstd` has shape `(2, 3, 1, 1)`.
- **`x`** (`Tensor`): Sum of `x1` and `x2`, $x$ in the formulas. The data layout can be ND. Empty and non-contiguous Tensors are supported. The shape and data type must be the same as those of the input `x1`.

## Constraints

- This API supports training and inference scenarios.
- This API supports single-operator mode and graph mode.
- Dimension boundary description:
  Each dimension in the shapes of the parameters `x1`, `x2`, `gamma`, `y1`, `y2`, `rstd`, and `x` must not be greater than the maximum value of `torch.int32`, 2147483647.
- Boundary value scenarios:
  - When the input is Inf, the output is Inf.
  - When the input is NaN, the output is NaN.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu

    input_x1 = torch.randn([20, 10, 64], dtype=torch.float16).npu()
    input_x2 = torch.randn([20, 10, 64], dtype=torch.float16).npu()
    input_gamma = torch.randn([64], dtype=torch.float16).npu()

    y1, y2, rstd, x = torch_npu.npu_add_rms_norm_cast(input_x1, input_x2, input_gamma)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    class EagerMode(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_x1, input_x2, input_gamma):
            y1, y2, rstd, x = torch_npu.npu_add_rms_norm_cast(input_x1, input_x2, input_gamma)
            return y1, y2, rstd, x

    def dynamo_mode_api(input_x1, input_x2, input_gamma):
        model = EagerMode().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        output = model(input_x1, input_x2, input_gamma)
        return output
    input_x1 = torch.randn([20, 10, 64], dtype=torch.float16).npu()
    input_x2 = torch.randn([20, 10, 64], dtype=torch.float16).npu()
    input_gamma = torch.randn([64], dtype=torch.float16).npu()
    outputs = dynamo_mode_api(input_x1, input_x2, input_gamma)
    ```
