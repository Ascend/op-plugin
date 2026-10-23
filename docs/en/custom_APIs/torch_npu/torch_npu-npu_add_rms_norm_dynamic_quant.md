# torch_npu.npu_add_rms_norm_dynamic_quant

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

## Function

- **API function**:

    RMSNorm is a normalization operation commonly used in foundation models. Compared with LayerNorm, it removes the mean subtraction. DynamicQuant performs symmetric dynamic quantization on the input Tensor. AddRmsNormDynamicQuant fuses the Add operation before RMSNorm and the one or two DynamicQuant operators applied to the RMSNorm output to reduce data movement.

- **Formulas**:

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

  In the formulas, row\_max indicates the maximum value of each row.

## Prototype

```python
torch_npu.npu_add_rms_norm_dynamic_quant(x1, x2, gamma, *, smooth_scale1=None, smooth_scale2=None, beta=None, epsilon=1e-6, output_mask=None, y_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`x1`** (`Tensor`): Required. First input for the Add computation, $x1$ in the formulas. The data layout can be ND. Empty Tensors are not supported. Non-contiguous Tensors are supported. The data type can be `torch.float16` or `torch.bfloat16`. Tensors with 2 to 8 dimensions are supported.
- **`x2`** (`Tensor`): Required. Second input for the Add computation, $x2$ in the formulas. The data layout can be ND. Empty Tensors are not supported. Non-contiguous Tensors are supported. The data type can be `torch.float16` or `torch.bfloat16`. The shape must be the same as that of `x1`.
- **`gamma`** (`Tensor`): Required. Scaling factor (weight) for RMSNorm, $gamma$ in the formulas. The data layout can be ND. Empty Tensors are not supported. Non-contiguous Tensors are supported. The data type is the same as that of `x1`. It must be 1D, and the number of elements must be the same as the size of the last dimension of `x1`.
- **`*`**: Syntax separator used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional parameters and are order-independent. They must be assigned using key-value pairs, and default values are used if they are not assigned.
- **`smooth_scale1`** (`Tensor`): Optional. Smooth scaling factor for the first quantization branch, $smoothScale1Optional$ in the formulas. The data type is the same as that of `x1`. It must be 1D, and the number of elements must be the same as the size of the last dimension of `x1`. The default value is `None`. When it is `None`, no smoothing is performed in the quantization branch.
- **`smooth_scale2`** (`Tensor`): Optional. Smooth scaling factor for the second quantization branch, $smoothScale2Optional$ in the formulas. The data type is the same as that of `x1`. It must be 1D, and the number of elements must be the same as the size of the last dimension of `x1`. The default value is `None`. When it is `None`, no smoothing is performed in the quantization branch.
- **`beta`** (`Tensor`): Optional. Bias term for RMSNorm, $beta$ in the formulas. The data type is the same as that of `x1`. It must be 1D, and the number of elements must be the same as the size of the last dimension of `x1`. The default value is `None`. When it is `None`, no bias is added.
- **`epsilon`** (`float`): Optional. Value added to the denominator to ensure numerical stability, $epsilon$ in the formulas. The default value is `1e-6`.
- **`output_mask`** (`bool[2]`): Optional. Boolean array of length 2, used to control whether to compute the two quantization outputs, $outputMask$ in the formulas. output_mask[0] controls the first quantization output (y1, scale1), and output_mask[1] controls the second quantization output (y2, scale2).
- **`y_dtype`** (`int`): Optional. Quantization output data type of y1 and y2. The supported values are None, `torch.int8`, `torch.quint4x2` or `torch_npu.int4`, `torch.float8_e5m2`, `torch.float8_e4m3fn`, and `torch_npu.hifloat8`. In the `torch.quint4x2` or `torch_npu.int4` scenario, the last dimension of `x1` must be divisible by 8. The default value is None.

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The supported values are None, `torch.int8`, and `torch.quint4x2` or `torch_npu.int4`.
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT products</term>: The supported values are None, `torch.int8`, `torch.quint4x2` or `torch_npu.int4`, `torch.float8_e5m2`, `torch.float8_e4m3fn`, and `torch_npu.hifloat8`.
  <!-- end id5 -->

## Return Values

- **`y1`** (`Tensor`): Output Tensor after the first dynamic quantization, $y1Out$ in the formulas. It is a valid output when `output_mask` is None or output_mask[0] is True. The data type can be `torch.int8`, `torch_npu.int4`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`. The data type is specified by `y_dtype`. If `y_dtype` indicates `torch_npu.int4`, the output dtype is `torch.int32` and the last dimension of the shape is 1/8 of the last dimension of the input `x1`, while other dimensions are the same as those of the input `x1`. If `y_dtype` indicates `torch_npu.hifloat8`, the output dtype is `torch.uint8` and the shape is the same as that of the input `x1`. If `y_dtype` is neither `torch_npu.int4` nor `torch_npu.hifloat8`, the output dtype is the same as the data type indicated by `y_dtype` and the shape is the same as that of the input `x1`. When output_mask[0] is False, no output is produced and an empty Tensor is returned.

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.int8` or `torch_npu.int4`.
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch_npu.int4`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`.
  <!-- end id7 -->

- **`y2`** (`Tensor`): Output Tensor after the second dynamic quantization, $y2Out$ in the formulas. It is a valid output when `output_mask` is None and `smooth_scale2` is not None, or output_mask[1] is True. The data type and shape are the same as the valid output of `y1`. When output_mask[1] is False, no output is produced and an empty Tensor is returned.

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.int8` or `torch_npu.int4`.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch_npu.int4`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`.
  <!-- end id9 -->

- **`x_out`** (`Tensor`): Result of the Add computation, $x$ in the formulas. The data type and shape are the same as those of the input `x1`.
- **`scale1`** (`Tensor`): Scaling coefficient of the first dynamic quantization, $scale1Out$ in the formulas. The conditions for a valid output are the same as those of `y1`. The data type is `torch.float32`, and the shape is the shape of `x1` without the last dimension. When output_mask[0] is False, no output is produced and an empty Tensor is returned.
- **`scale2`** (`Tensor`): Scaling coefficient of the second dynamic quantization, $scale2Out$ in the formulas. The conditions for a valid output are the same as those of `y2`. The data type is `torch.float32`, and the shape is the shape of `x1` without the last dimension. When output_mask[1] is False, no output is produced and an empty Tensor is returned.

## Constraints

- When `output_mask` is not empty and `smooth_scale1` has a value, output_mask[0] must be True. When `smooth_scale2` has a value, output_mask[1] must be True.
- When `output_mask` is not empty, output_mask[0] and output_mask[1] cannot both be False.
- When `output_mask` is empty, `smooth_scale2` cannot be the only one present.

## Examples

- Single-operator call

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

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair
    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, gamma, smooth1, smooth2, beta, epsilon, output_mask):
            return torch_npu.npu_add_rms_norm_dynamic_quant(x1, x2, gamma, smooth_scale1=smooth1, smooth_scale2=smooth2, beta=beta, epsilon=epsilon, output_mask=output_mask)

    x1 = torch.randn(2, 3, 32, dtype=torch.float16, device='npu')
    x2 = torch.randn(2, 3, 32, dtype=torch.float16, device='npu')
    gamma = torch.ones(32, dtype=torch.float16, device='npu')
    beta = torch.zeros(32, dtype=torch.float16, device='npu')
    smooth_scale1 = torch.ones(32, dtype=torch.float16, device='npu')
    smooth_scale2 = torch.ones(32, dtype=torch.float16, device='npu')
    epsilon = 1e-6
    output_mask = [True, True]

    model = NetModel()
    model.to('npu')
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=True)
    y1, y2, x_out, scale1, scale2 = model(x1, x2, gamma, smooth_scale1, smooth_scale2, beta, epsilon, output_mask)
    y1 = y1.cpu()
    y2 = y2.cpu()
    x_out = x_out.cpu()
    scale1 = scale1.cpu()
    scale2 = scale2.cpu()
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
