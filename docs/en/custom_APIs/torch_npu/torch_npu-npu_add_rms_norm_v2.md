# torch_npu.npu_add_rms_norm_v2

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- **API function**: AddRmsNorm is a fusion of the Add operator and the RmsNorm operator. RmsNorm is a normalization operation commonly used in foundation models. Compared with LayerNorm, it removes the mean subtraction.

- **Formulas**:

  $$
  x_i=x1_i+x2_i
  $$

  $$
  \operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

## Prototype

```python
torch_npu.npu_add_rms_norm_v2(x1, x2, gamma, epsilon=1e-6) -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. It corresponds to $x1$ in the formulas and is the first input for the Add computation and the final normalized output. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The shape can have 1 to 8 dimensions. The data layout can be ND. Empty tensors are not supported.
- **`x2`** (`Tensor`): Required. It corresponds to $x2$ and $x$ in the formulas and is the second input for the Add computation and the result of the Add computation. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The shape must be the same as that of `x1`. The data layout can be ND. Empty tensors are not supported.
- **`gamma`** (`Tensor`): Required. It corresponds to $gamma$ in the formulas and is the scaling factor (weight) for RmsNorm. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The shape must be the same as the dimensions of `x1` to be normalized. The data layout can be ND. Empty tensors are not supported.
- **`epsilon`** (`float`): Optional. It corresponds to $epsilon$ in the formulas. The data type can be `double`. The default value is `1e-6`.

## Return Values

**`rstd`** (`Tensor`): Corresponds to $Rms(x)$ in the formulas. It is an intermediate result of the normalization and is used for backward computation. The data type can be `torch.float32` only. The number of dimensions must be the same as that of `x1`. The dimensions that do not require normalization have the same sizes as the corresponding dimensions of `x1`, and the dimensions that require normalization have a size of `1`. Examples of the relationship among the shapes of `x1`, `gamma`, and `rstd` are as follows:

  - If `x1` has shape `(2, 3, 4, 8)` and `gamma` has shape `(8,)`, `rstd` has shape `(2, 3, 4, 1)`.
  - If `x1` has shape `(2, 3, 4, 8)` and `gamma` has shape `(4, 8)`, `rstd` has shape `(2, 3, 1, 1)`.

## Constraints

- This API supports single-operator mode and TorchAir graph mode.
- The outputs overwrite the original `x1` and `x2`.
- In single-operator mode, the data types of `x2` and `gamma` must be the same as that of `x1`.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu

    x1 = torch.randn((32,128), dtype=torch.float32).npu()
    x2 = torch.randn((32,128), dtype=torch.float32).npu()
    gamma = torch.randn((128), dtype=torch.float32).npu()
    rstd = torch_npu.npu_add_rms_norm_v2(x1, x2, gamma, epsilon=1e-5)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair
    import numpy as np

    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, gamma, eps):
            return torch_npu.npu_add_rms_norm_v2(x1, x2, gamma, epsilon=eps)
    # generate inputs
    x1 = torch.randn((32,128), dtype=torch.float32).npu()
    x2 = torch.randn((32,128), dtype=torch.float32).npu()
    gamma = torch.randn((128), dtype=torch.float32).npu()
    model = NetModel()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    rstd = model(x1, x2, gamma, 1e-5)
    ```
