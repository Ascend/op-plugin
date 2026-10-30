# torch_npu.npu_dynamic_mx_quant_with_dual_axis

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->

## Function

- API function: Performs MX quantization with the target data type `float4` (`torch_npu.float4_e1m2fn_x2`, `torch_npu.float4_e2m1fn_x2`) or `float8` (`torch.float8_e4m3fn`, `torch.float8_e5m2`) simultaneously on the -1 and -2 axes of the input Tensor. On the -1 and -2 axes, the quantization scales mxscale1 and mxscale2 are calculated for every 32 values and used as the corresponding parts of the output tensors mxscale1 and mxscale2. Then, all elements in the two groups are divided by the corresponding quantization scale and converted to the corresponding dst_type according to round_mode to obtain the quantization results y1 and y2.
  - Axis merging: In the operator implementation, all axes before the -2 axis (exclusive) are merged. That is, for an input Tensor with shape `(d_0, d_1, ..., d_{n-3}, d_{n-2}, d_{n-1})`, the dimensions before the -2 axis `(d_0, d_1, ..., d_{n-3})` are merged into one dimension, which is equivalent to reshaping the input to `(d_0 \times d_1 \times ... \times d_{n-3}, d_{n-2}, d_{n-1})` before quantization.
- Formulas:
  - Scenario 1, when scale_alg is 0, that is, the implementation of the OCP Microscaling Formats (Mx) Specification:
    - The input x is grouped along the -1 axis into groups of 32 values. A group of 32 values $\{\{V_i\}_{i=1}^{32}\}$ is quantized to $\{mxscale1, \{P_i\}_{i=1}^{32}\}$:

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax
      $$

      $$
      mxscale1 = 2^{shared\_exp}
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
      $$

    - At the same time, the input x is grouped on the -2 axis by 32 values. A group of 32 values $\{\{V_j\}_{j=1}^{32}\}$ is quantized to $\{mxscale2, \{P_j\}_{j=1}^{32}\}$:

      $$
      shared\_exp = floor(log_2(max_j(|V_j|))) - emax
      $$

      $$
      mxscale2 = 2^{shared\_exp}
      $$

      $$
      P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
      $$

    - The quantized $P_i$ on the -1 axis forms the output y1 at the positions of $V_i$, and mxscale1 forms the output mxscale1 by grouping on the corresponding -1 axis dimension. The quantized $P_j$ on the -2 axis forms the output y2 at the positions of $V_j$, and mxscale2 forms the output mxscale2 by grouping on the corresponding -2 axis dimension.
    - emax: Exponent bit of the maximum normal number of the corresponding data type. The mapping is as follows.

      | dst_type | emax |
      | --- | --- |
      | `torch_npu.float4_e2m1fn_x2` | 2 |
      | `torch_npu.float4_e1m2fn_x2` | 0 |
      | `torch.float8_e4m3fn` | 8 |
      | `torch.float8_e5m2` | 15 |

  - Scenario 2, when scale_alg is 1, only float8 is involved (cuBLAS Scale calculation algorithm):
    - -1 axis quantization: The input x is grouped along the -1 axis into groups of 32 values. Each group has a length of 32. A block scaling factor $S_{fp32}^b$ is calculated separately for each group, and all elements in the group are mapped to the target low-precision type float8 using the same $S_{fp32}^b$. If the last group has fewer than 32 elements, the missing values are treated as 0 and the group is processed as a complete group. Find the maximum absolute value in the group:

      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{32})
      $$

      Map fp32 to the range representable by the target data type float8, where $Amax(DType)$ is the maximum value representable by the target precision:

      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$

      Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in fp8 format. Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from $S_{fp32}^b$. To prevent overflow during quantization, round the exponent up and keep it within the range representable by fp8:

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, } E_{int}^b < 254, \text{ and } M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a subnormal number and } M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{otherwise} \end{cases}
      $$

      Calculate the block scaling factor $S_{ue8m0}^b=2^{E_{int}^b}$ and the block conversion factor $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$. For each element in the group, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantization result output on the -1 axis is $\left(S^b, [d^i]_{i=1}^{32}\right)$, where $S^b$ is the block scaling factor (that is, $S_{ue8m0}^b$).
    - -2 axis quantization: At the same time, the input x is grouped on the -2 axis by 32 values. The same cuBLAS Scale calculation algorithm as that for the -1 axis is used. The block scaling factor is calculated independently for each group and quantization is performed. The quantization result output on the -2 axis is $\left(S^b, [d^j]_{j=1}^{32}\right)$.
    - The -1 axis quantization results form the output y1, and the corresponding block scaling factors form the output mxscale1. The -2 axis quantization results form the output y2, and the corresponding block scaling factors form the output mxscale2.

  - Scenario 3, when scale_alg is 2, only the `torch_npu.float4_e2m1fn_x2` type is involved:
    - When dst_type_max is 0.0/6.0/7.0, the input x is grouped on the -1 and -2 axes by 32 values and dynamically quantized to $\{mxscale, \{P\}\}$:

      $$
      shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{if the first 1/2 high bits of the mantissa are 1 and the mantissa is not all 0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{otherwise} \end{cases}
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space 32
      $$

    - When dst_type_max is not one of the preceding special values, the input x is grouped on the -1 and -2 axes by 32 values. Each group has a length of 32. The same block scaling algorithm as that in scenario 2 is used: a block scaling factor $S_{fp32}^b$ is calculated separately for each group and mapped to the target low-precision type `torch_npu.float4_e2m1fn_x2`. When dst_type_max is 0, $Amax(DType)$ is the maximum value representable by the target precision; when it is not 0, $Amax(DType)$ is the value passed in dst_type_max:

      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$

      The exponent round-up rule is:

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, } E_{int}^b < 254, \text{ and } M_{fixp}^b > 0 \\ E_{int}^b, & \text{otherwise} \end{cases}
      $$

      The remaining steps (calculating $S_{ue8m0}^b$, $R_{fp32}^b$, and $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$) are the same as those in scenario 2.
    - The -1 axis quantization results form the output y1, and the corresponding block scaling factors form the output mxscale1. The -2 axis quantization results form the output y2, and the corresponding block scaling factors form the output mxscale2.

## Prototype

```python
torch_npu.npu_dynamic_mx_quant_with_dual_axis(input, *, round_mode="rint", dst_type=296, scale_alg=0,
                                              dst_type_max=0.0) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Input x. The number of dimensions can be 2 to 7. The data layout can be ND. The data type can be `torch.float16` or `torch.bfloat16`. Non-contiguous Tensors are supported. When dst_type is float4 (that is, the types corresponding to 296 and 297), the last dimension of input must be even.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`round_mode`** (`str`): Optional. Data conversion mode, $round\_mode$ in the formulas. The default value is "rint".
  - When dst_type is `torch.float8_e5m2` or `torch.float8_e4m3fn`, only "rint" is supported.
  - When dst_type is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, "rint", "floor", and "round" are supported.
- **`dst_type`** (`int`): Optional. Data type of y1 and y2 after conversion. The default value is 296. The supported values are:
  - 23: `torch.float8_e5m2`.
  - 24: `torch.float8_e4m3fn`.
  - 296: `torch_npu.float4_e2m1fn_x2`.
  - 297: `torch_npu.float4_e1m2fn_x2`.
- **`scale_alg`** (`int`): Optional. Calculation method of mxscale1 and mxscale2. The value can be 0, 1, or 2. The default value is 0.
  - 0: Scenario 1 (OCP MX specification implementation). Both float8 and float4 are supported.
  - 1: Scenario 2 (cuBLAS Scale calculation algorithm). Only float8 is supported.
  - 2: Scenario 3 (fp4 dynamic quantization). Only `torch_npu.float4_e2m1fn_x2` is supported.
- **`dst_type_max`** (`float`): Optional. Value of maxType, $Amax(DType)$ in the formulas. The default value is 0.0. This value can be set only when dst_type is `torch_npu.float4_e2m1fn_x2` and scale_alg is 2. The supported values are 0.0 and 6.0 to 12.0. When the value is 0.0, Amax(DType) is the maximum value of the quantized result data type. When the value is 6.0 to 12.0, Amax(DType) is the passed value.

## Return Values

- **`y1`** (`Tensor`): Result of quantizing the -1 axis of the input x. The shape is the same as that of the input x. When dst_type is float8 (23, 24), the data type is `torch.float8_e5m2` or `torch.float8_e4m3fn`. When dst_type is float4 (296, 297), the actual returned data type is `torch.uint8`, and the last dimension is half the last dimension of the input x (each `torch.uint8` element packs two float4 values). You need to unpack it to view the specific values, such as `y1.view(torch_npu.float4_e2m1fn_x2)`. Non-contiguous Tensors are not supported.
- **`mxscale1`** (`Tensor`): Quantization scale corresponding to each group on the -1 axis, $mxscale1$ in the formulas. The logical data type is `torch.float8_e8m0fnu`, and the actual returned data type is `torch.uint8`. You need to convert it to view the specific values, such as `mxscale1.view(torch.float8_e8m0fnu)`. The shape is the value of the -1 axis of the input x divided by 32, rounded up, padded to an even number (the pad value is 0), and finally appended with a dimension of size 2. Non-contiguous Tensors are not supported.
- **`y2`** (`Tensor`): Result of quantizing the -2 axis of the input x. The shape is the same as that of the input x. The data type is the same as that of y1 (for float4, the actual returned data type is `torch.uint8`, and the last dimension is half the last dimension of the input x). Non-contiguous Tensors are not supported.
- **`mxscale2`** (`Tensor`): Quantization scale corresponding to each group on the -2 axis, $mxscale2$ in the formulas. The logical data type is `torch.float8_e8m0fnu`, and the actual returned data type is `torch.uint8`. You need to convert it to view the specific values. The shape is the value of the -2 axis of the input x divided by 32, rounded up, padded to an even number (the pad value is 0), and finally appended with a dimension of size 2. The output must be interleaved for every two rows of data. Non-contiguous Tensors are not supported.

## Constraints

- This API supports training and inference scenarios.
- This API supports single-operator mode and graph mode.
- Deterministic computation: The deterministic implementation is used by default.
- The parameter combination constraints for dst_type, scale_alg, round_mode, and dst_type_max are as follows.

| dst_type | scale_alg | round_mode | dst_type_max |
| --- | --- | --- | --- |
| `torch.float8_e5m2` | 0, 1 | "rint" | Cannot be set (only the default value 0.0) |
| `torch.float8_e4m3fn` | 0, 1 | "rint" | Cannot be set (only the default value 0.0) |
| `torch_npu.float4_e2m1fn_x2` | 0, 2 | "rint"/"floor"/"round" | Takes effect only when scale_alg=2. 0.0 and 6.0-12.0 are supported. |
| `torch_npu.float4_e1m2fn_x2` | 0 | "rint"/"floor"/"round" | Cannot be set (only the default value 0.0) |

- Shape constraints for input, mxscale1, and mxscale2 are as follows:
  - rank(mxscale1) = rank(input) + 1.
  - rank(mxscale2) = rank(input) + 1.
  - mxscale1.shape[-2] = (ceil(input.shape[-1] / 32) + 2 - 1) // 2.
  - mxscale2.shape[-3] = (ceil(input.shape[-2] / 32) + 2 - 1) // 2.
  - mxscale1.shape[-1] = 2.
  - mxscale2.shape[-1] = 2.
  - Other dimensions are the same as those of the input.
  - For example, if the shape of the input is [B, M, N] and the target data type is float8, the shapes of y1 and y2 are [B, M, N], the shape of mxscale1 is [B, M, (ceil(N/32)+2-1)/2, 2], and the shape of mxscale2 is [B, (ceil(M/32)+2-1)/2, N, 2].

## Examples

- Single-operator call, float8 quantization (`scale_alg` uses the default value 0)

    ```python
    import torch
    import torch_npu
    
    # The shape of input is [1, 4], and the values are (0, 8, 64, 512).
    input = torch.tensor([[0, 8, 64, 512]], dtype=torch.bfloat16).npu()
    y1, mxscale1, y2, mxscale2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        input, dst_type=torch.float8_e4m3fn)
    print(y1)
    print(mxscale1.shape, mxscale1.dtype)
    print(mxscale1.cpu().view(torch.float8_e8m0fnu))
    print(y2.shape, y2.dtype)
    print(mxscale2.shape, mxscale2.dtype)
    print(mxscale2.cpu().view(torch.float8_e8m0fnu))
    ```

    The output is as follows:

    ```text
    tensor([[  0.,   4.,  32., 256.]], device='npu:0', dtype=torch.float8_e4m3fn)
    torch.Size([1, 1, 2]) torch.uint8
    tensor([[[2., 0.]]], dtype=torch.float8_e8m0fnu)
    torch.Size([1, 4]) torch.float8_e4m3fn
    torch.Size([1, 4, 2]) torch.uint8
    tensor([[[0.0000, 0.0000],
             [0.0312, 0.0000],
             [0.2500, 0.0000],
             [2.0000, 0.0000]]], dtype=torch.float8_e8m0fnu)
    ```

- Single-operator call, float4 quantization (`scale_alg`=2)

    ```python
    import torch
    import torch_npu
    
    # When dst_type is float4, the last dimension of input must be a multiple of 2.
    input = torch.randn((64, 256), dtype=torch.bfloat16).npu()
    y1, mxscale1, y2, mxscale2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        input, dst_type=torch_npu.float4_e2m1fn_x2, scale_alg=2, dst_type_max=6.0)
    print(y1.shape, y1.dtype)
    print(mxscale1.shape)
    print(mxscale2.shape)
    ```

    The output is as follows:

    ```text
    torch.Size([64, 128]) torch.uint8
    torch.Size([64, 4, 2])
    torch.Size([1, 256, 2])
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch_npu.npu_dynamic_mx_quant_with_dual_axis(x, dst_type=torch.float8_e4m3fn)

    model = Model().npu()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    input = torch.randn((64, 256), dtype=torch.bfloat16).npu()
    y1, mxscale1, y2, mxscale2 = model(input)
    print(y1.shape, y1.dtype)
    print(mxscale1.shape)
    print(y2.shape, y2.dtype)
    print(mxscale2.shape)
    ```

    The output is as follows:

    ```text
    torch.Size([64, 256]) torch.float8_e4m3fn
    torch.Size([64, 4, 2])
    torch.Size([64, 256]) torch.float8_e4m3fn
    torch.Size([1, 256, 2])
    ```
