# (beta) torch_npu.npu_rms_norm

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

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
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

## Function

- API function: RMSNorm is a normalization operation commonly used in foundation models. Compared with LayerNorm, it removes the mean subtraction.
- Formulas:

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i
  $$

  $$
  \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

## Prototype

```python
torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor) 
```

## Parameters

- **`self`** (`Tensor`): Required. Input for the normalization computation. It corresponds to $x$ in the formulas. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The shape can have 2 to 8 dimensions. The data layout can be ND. Non-contiguous Tensors and empty Tensors are supported.
- **`gamma`** (`Tensor`): Required. Scaling factor (weight) for the normalization computation. It corresponds to $g$ in the formulas. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The shape can have 2 to 8 dimensions. The data layout can be ND. The shape must satisfy gamma_shape = self_shape[n:], n < self_shape.dims(), and is usually the last dimension of `self`. Non-contiguous Tensors and empty Tensors are supported.
- **`epsilon`** (`float`): Optional. Used to prevent division by zero. It corresponds to `eps` in the formulas. The data type is `double`. The default value is 1e-6.

## Return Values

- **`yOut`** (`Tensor`): Final output after normalization, corresponding to the final output $RmsNorm(x)$ in the formulas. The data type and shape are the same as those of the input `self`. Non-contiguous Tensors and empty Tensors are supported.
- **`rstdOut`** (`Tensor`): Reciprocal of the standard deviation after normalization. It is an intermediate result of rms_norm, corresponding to the reciprocal of $Rms(x)$ in the formulas, and is used for backward computation. The data type is `torch.float32`. The shape is the same as the first several dimensions of the shape of the input parameter `self`, where the first several dimensions refer to the dimensions of `self` minus the dimensions of `gamma`, indicating the dimensions that do not require normalization. Non-contiguous Tensors and empty Tensors are supported.

## Constraints

- This API supports single-operator mode.

<!-- npu="950" id6 -->
- <term>Ascend 950PR&950DT products</term>: This API also supports TorchAir graph mode.
<!-- end id6 -->
<!-- npu="310p" id7 -->
- <term>Atlas inference products</term>: The tail axis length of the `self` and `gamma` inputs must be greater than or equal to 32 Bytes.
<!-- end id7 -->
- Boundary value scenarios:

  <!-- npu="310p" id8 -->
  - <term>Atlas inference products</term>: The input cannot contain Inf or NaN.
  <!-- end id8 -->
  <!-- npu="950,A3,910b" id9 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, and <term>Ascend 950PR&950DT products</term>: When the input is Inf, the output is Inf. When the input is NaN, the output is NaN.
  <!-- end id9 -->

- Data types supported by each product and their mappings:

  <!-- npu="950,A3,910b" id10 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, and <term>Ascend 950PR&950DT products</term>:

    | `self` Data Type | `gamma` Data Type | `yOut` Data Type | `rstdOut` Data Type |
    | -------- | -------- | -------- | -------- |
    | `torch.float16` | `torch.float32` | `torch.float16` | `torch.float32` |
    | `torch.bfloat16` | `torch.float32` | `torch.bfloat16` | `torch.float32` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float32` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` |
    | `torch.float32` | `torch.float32`  | `torch.float32` | `torch.float32` |
  <!-- end id10 -->
  <!-- npu="910,310p" id11 -->
  - <term>Atlas inference products</term> and <term>Atlas training products</term>:

    | `self` Data Type | `gamma` Data Type | `yOut` Data Type | `rstdOut` Data Type |
    | -------- | -------- | -------- | -------- |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float32` |
    | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` |
  <!-- end id11 -->

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu

    x = torch.randn((24, 1, 128), dtype=torch.bfloat16).npu()
    w = torch.randn((128), dtype=torch.bfloat16).npu()
    y, rstd = torch_npu.npu_rms_norm(x, w, epsilon=1e-5)
    ```

<!-- npu="950" id12 -->
- Graph mode call: This example is supported only on <term>Ascend 950PR&950DT products</term>.

    ```python
    import torch
    import torch_npu
    import torchair
    import numpy as np


    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, gamma, eps):
            return torch_npu.npu_rms_norm(x, gamma, epsilon=eps)
    # generate inputs
    x = torch.randn((32,128), dtype=torch.float32).npu()
    gamma = torch.randn((128), dtype=torch.float32).npu()
    model = NetModel()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    y, rstd = model(x, gamma, 1e-5)
    ```
<!-- end id12 -->
