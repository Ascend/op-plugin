# torch_npu.npu_add_rms_norm

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->

## Function

- Fuses Add computation with RMSNorm normalization, commonly used in foundation models to normalize tensors after residual connections.
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

- **`x1`** (`Tensor`): Required. First input for Add computation, $x1$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`. The shape must have 1 to 8 dimensions.
- **`x2`** (`Tensor`): Required. Second input for Add computation, $x2$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`. The shape must have 1 to 8 dimensions.
- **`gamma`** (`Tensor`): Required. Scaling factor (weight) for RMSNorm, $gamma$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type must be identical to that of `x1`. The shape must be identical to the trailing dimensions of `x1`, which correspond to the dimensions to be normalized.
- **`epsilon`** (`float`): Optional. Value added to the denominator to ensure numerical stability, $epsilon$ in the formulas. The default value is `1e-6`.

## Return Values

- **`yOut`** (`Tensor`): Final output, $RMSNorm(x)$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type and shape must be identical to those of the input `x1`.

- **`rstdOut`** (`Tensor`): Reciprocal of the normalized root mean square, reciprocal of $RMS(x)$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type can be `torch.float32`. The shape must be identical to the leading dimensions of `x1`. The leading dimensions refer to the dimensions that do not require normalization. Examples of the relationship among the shapes of `rstdOut`, `x1`, and `gamma` are as follows:
  - If `x1` has shape `(2, 3, 4, 8)` and `gamma` has shape `(8,)`, `rstdOut` has shape `(2, 3, 4, 1)`.
  - If `x1` has shape `(2, 3, 4, 8)` and `gamma` has shape `(4, 8)`, `rstdOut` has shape `(2, 3, 1, 1)`.

- **`xOut`** (`Tensor`): Result of Add computation, $x$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The data type and shape must be identical to those of the input `x1`.

## Constraints

- This API supports single-operator mode and TorchAir graph mode.
- In single-operator mode, the data types of the input parameters `x2` and `gamma` must be the same as that of `x1`.
- Boundary value scenarios:
  - When the input is `Inf`, the output is `Inf`.
  - When the input is `NaN`, the output is `NaN`.

<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>:
  - The input parameters `x1`, `x2`, and `gamma` do not support the `torch.bfloat16` data type.
  - The parameter `rstdOut` is invalid in current product usage scenarios.
<!-- end id5 -->

## Example

- Single-operator call
    
    ```python
    import torch
    import torch_npu

    x1 = torch.randn((32,128), dtype=torch.float32).npu()
    x2 = torch.randn((32,128), dtype=torch.float32).npu()
    gamma = torch.randn((128), dtype=torch.float32).npu()
    y, rstd, x = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=1e-6)
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
            return torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=eps)
    # generate inputs
    x1 = torch.randn((32,128), dtype=torch.float32).npu()
    x2 = torch.randn((32,128), dtype=torch.float32).npu()
    gamma = torch.randn((128), dtype=torch.float32).npu()
    model = NetModel()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    y, rstd, x = model(x1, x2, gamma, 1e-6)
    ```
