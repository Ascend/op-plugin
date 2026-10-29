# torch_npu.npu_rms_norm_dynamic_mx_quant

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- **API function**: Fuses the RmsNorm operator with the DynamicMxQuant operator. The input is first normalized by RmsNorm, the RmsNorm result is added to beta, and then dynamic MX quantization is performed to reduce data movement operations. RmsNorm is a normalization operation commonly used in foundation models. Compared with LayerNorm, it removes the mean subtraction.

- **Formulas**:
    1. RmsNorm computation

        $$
        \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
        $$

    2. Add beta to the RmsNorm result

        $$
        y=\operatorname{RmsNorm}(x)+beta
        $$

    3. DynamicMxQuant quantization
        - **Scenario 1: When scale_alg is 0, dynamic MX quantization of fp4 and fp8 is supported.**

            The RmsNorm output y is grouped along the last axis into groups of k=blocksize values. A group of k values $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale,\{P_i\}_{i=1}^{k}\}$, 1<=i<=k, blocksize=32.

            $$
            shared\_exp = floor(log_2(max_i(|V_i|))) - emax
            $$

            $$
            mxscale = 2^{shared\_exp}
            $$

            $$
            P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
            $$

            - The quantized $P_i$ forms the output y at the positions of $V_i$ in the original x Tensor, and mxscale forms the output mxscale_out by grouping on the corresponding axis.

            - In the formula, emax is the exponent bit of the maximum normal number of the corresponding data type. The mapping is as follows.

                | dst_type | emax |
                | --- | --- |
                | `torch.float4_e2m1fn_x2` | 2 |
                | `torch_npu.float4_e1m2fn_x2` | 0 |
                | `torch.float8_e4m3fn` | 8 |
                | `torch.float8_e5m2` | 15 |

        - **Scenario 2: When scale_alg is 1, dynamic MX quantization of fp8 is supported.**
            - The long vector is divided into blocks, each with length k. A block scaling factor $S_{fp32}^b$ is calculated separately for each block, and all elements in the block are mapped to the target low-precision type fp8 by using the same $S_{fp32}^b$. If the last block has fewer than k elements, the missing values are treated as 0 and the block is processed as a complete block.
            - Find the maximum absolute value in the block:

                $$
                Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
                $$

            - Map FP32 to the range representable by the target data type fp8, where Amax(DType) is the maximum value representable by the target precision:

                $$
                S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
                $$

            - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in fp8 format.
            - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scaling factor $S_{fp32}^b$.
            - To prevent overflow during quantization, round the exponent up and keep it within the range representable by fp8:

                $$
                E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, } E_{int}^b < 254, \text{ and } M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a subnormal number and } M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{otherwise} \end{cases}
                $$

            - Calculate the block scaling factor:

                $S_{ue8m0}^b=2^{E_{int}^b}$

            - Calculate the block conversion factor:

                $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$

            - Apply it to the final quantization step. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The final quantization result is $(S^b, [d^i]_{i=1}^{k})$, where $S^b$ is the block scaling factor, that is, $S_{ue8m0}^b$, and $[d^i]_{i=1}^{k}$ is the quantized data in the block.

## Prototype

```python
torch_npu.npu_rms_norm_dynamic_mx_quant(x, gamma, *, beta=None, epsilon=1e-06, scale_alg=0, round_mode="rint", dst_type=torch.float4_e2m1fn_x2) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input for the normalization computation, $x$ in the formulas. The shape can have 1 to 7 dimensions. The data type can be `torch.float16` or `torch.bfloat16`. The data layout must be ND. Empty Tensors and non-contiguous Tensors are supported. When the output `y` type is `torch.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, that is, when dst_type is 296 or 297, the last dimension of `x` must be even.
- **`gamma`** (`Tensor`): Required. Scaling factor for RmsNorm, $gamma$ in the formulas. The shape is 1D and must be the same as the last dimension of `x`. The data type must be the same as that of `x` or be `torch.float32`. The data layout can be ND. Empty Tensors and non-contiguous Tensors are supported.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`beta`** (`Tensor`): Optional. Bias Tensor added to the RmsNorm result, $beta$ in the formulas. If present, the shape, data type, and data layout must be the same as those of `gamma`. Empty Tensors and non-contiguous Tensors are supported.
- **`epsilon`** (`float`): Optional. Value added to the denominator to ensure numerical stability, $epsilon$ in the formulas. The data type is `double`. The default value is `1e-06`.
- **`scale_alg`** (`int`): Optional. Calculation method of `mxscale_out`. The value can be 0 (default) or 1. When `dst_type` is 296 (`torch.float4_e2m1fn_x2`) or 297 (`torch_npu.float4_e1m2fn_x2`), only 0 is supported. When `dst_type` is 291 (`torch.float8_e5m2`) or 292 (`torch.float8_e4m3fn`), 0 and 1 are supported. The default value is 0.
  - Value 0: Scenario 1, indicating the OCP (Open Compute Project) implementation.
  - Value 1: Scenario 2, indicating the cuBLAS implementation.

- **`round_mode`** (`str`): Optional. Mode for casting the quantization result to the data type of the output `y`. When dst_type is 296 (`torch.float4_e2m1fn_x2`) or 297 (`torch_npu.float4_e1m2fn_x2`), the supported modes are {"rint", "floor", "round"}. When dst_type is 291 (`torch.float8_e5m2`) or 292 (`torch.float8_e4m3fn`), only "rint" is supported. The default value is "rint".
- **`dst_type`** (`int`): Optional. Data type of the output y. The input range is 291 (`torch.float8_e5m2`), 292 (`torch.float8_e4m3fn`), 296 (`torch.float4_e2m1fn_x2`, default), or 297 (`torch_npu.float4_e1m2fn_x2`).

## Return Values

- **`y`** (`Tensor`): Output result after RmsNorm normalization, addition with `beta`, and DynamicMxQuant quantization. Empty Tensors are supported. The shape and data layout are the same as those of the input `x`. The data type is specified by `dst_type`. When `dst_type` is 296 (`torch.float4_e2m1fn_x2`) or 297 (`torch_npu.float4_e1m2fn_x2`), the data type of y is actually `torch.uint8`.
- **`mxscale_out`** (`Tensor`): Quantization scale corresponding to each group. The logical data type is `torch.float8_e8m0fnu`, and the data layout must be ND. The actual data type is `torch.uint8`, and the shape can have 2 to 8 dimensions. On the last axis, the shape is the corresponding value of `x` divided by 32, rounded up, and padded to an even number with a padding value of 0. For details about the calculation, see Constraints.
- **`rstd_out`** (`Tensor`): Reciprocal of the standard deviation after RmsNorm normalization, used for the normalization operation and corresponding to the reciprocal of $Rms(x)$ in the formulas. The shape is the same as the leading dimensions of `x`, which are the dimensions that do not require normalization. The data layout must be ND. Example of the relationship among the shapes of `x`, `gamma`, and `rstd_out`: if `x` has shape (2,3,4,8) and `gamma` has shape (8,), rstd_out.shape=(2,3,4,1).

## Constraints

- This API supports training and inference scenarios.
- This API supports single-operator mode and graph mode.
- When the output `y` type is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, that is, when dst_type is 296 or 297, the last dimension of `x1` must be even.
- The quantization axis and the norm axis are both the last dimension of the input x.
- Shape constraints between the output `mxscale_out` and the input `x1`:
  - rank(mxscale_out) = rank(x1) + 1
  - mxscale_out.shape[-2] = (ceil(x1.shape[-1] / 32) + 2 - 1) / 2
  - mxscale_out.shape[-1] = 2
  - Other dimensions are the same as those of the input x1.
  - For example, if the input x1.shape = (100, 8, 64), mxscale_out.shape = (100, 8, 1, 2).

- **In single-operator mode**, the operator determines whether to output a valid rstd by reading the requires_grad attribute of the input Tensor (x). requires_grad is a standard attribute of PyTorch Tensors, and its default value is False. When requires_grad=False, the operator does not write rstd, and the API returns an empty Tensor with shape[0]. In this case, rstd is an invalid placeholder output.
- **The graph mode path** does not support backpropagation. rstd is fixed to a scalar Tensor with value 0 (shape[]), which does not contain valid data.

- **Boundary value scenarios**

  - When the input is Inf:
    - The output y is 0.
    - The output mxscale_out is 255, and the even pad value is 0.
    - The output rstd_out is 0.

  - When the input is NaN:
    - The output y is 0.
    - The output mxscale_out is 255, and the even pad value is 0.
    - The output rstd_out is NaN.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu

    x = torch.randn([8, 64], dtype=torch.float16, requires_grad=True).npu()
    gamma = torch.randn([64], dtype=torch.float16).npu()
    beta = torch.randn([64], dtype=torch.float16).npu()
    y_npu, mxscale_npu, rstd_npu = torch_npu.npu_rms_norm_dynamic_mx_quant(x, gamma, beta=beta, epsilon=1e-6, scale_alg=0, round_mode="rint", dst_type=torch_npu.float8_e5m2)
    y = y_npu.cpu()
    mxscale = mxscale_npu.cpu()
    rstd = rstd_npu.cpu()
    print(f"y dtype = {y.dtype}, y shape = {y.shape}")
    print(f"mxscale dtype = {mxscale.dtype}, mxscale shape = {mxscale.shape}")
    print(f"rstd dtype = {rstd.dtype}, rstd shape = {rstd.shape}")
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair
    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, gamma, beta):
            return torch_npu.npu_rms_norm_dynamic_mx_quant(x, gamma, beta=beta, epsilon=1e-6, scale_alg=0, round_mode="rint", dst_type=torch_npu.float8_e5m2)

    x = torch.randn([8, 64], dtype=torch.float16).npu()
    gamma = torch.randn([64], dtype=torch.float16).npu()
    beta = torch.randn([64], dtype=torch.float16).npu()
    model = NetModel()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    y_npu, mxscale_npu, rstd_npu = model(x, gamma, beta)
    y = y_npu.cpu()
    mxscale = mxscale_npu.cpu()
    rstd = rstd_npu.cpu()
    print(f"y dtype = {y.dtype}, y shape = {y.shape}")
    print(f"mxscale dtype = {mxscale.dtype}, mxscale shape = {mxscale.shape}")
    print(f"rstd dtype = {rstd.dtype}, rstd shape = {rstd.shape}")
    ```
