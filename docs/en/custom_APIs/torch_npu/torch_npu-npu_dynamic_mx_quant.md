# torch_npu.npu_dynamic_mx_quant

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- API function: Dynamic MX quantization with a target data type of float4 or float8. On the given axis `axis`, the quantization scale `mxscale` corresponding to each group of `block_size` values is calculated. Each value in the group is divided by `mxscale` and converted to the corresponding `dst_type` according to `round_mode` to obtain the quantization result `y`. When `dst_type` is `torch.float8_e5m2` or `torch.float8_e4m3fn`, different algorithms for calculating `mxscale` are specified by the value of `scale_alg`.

- Formulas:

  - Scenario 1, when `scale_alg` is 0:

    - The input `input` is grouped along the `axis` dimension into groups of k = `block_size` values. A group of k values $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale, \{P_i\}_{i=1}^{k}\}$, k = `block_size`.
    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    mxscale = 2^{shared\_exp}\\
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space block\_size\\
    $$

    - The quantized $P_i$ forms the output `y` at the positions of $V_i$, and `mxscale` forms the output `mxscale` by grouping on the corresponding `axis` dimension.

    - emax: Exponent bit of the maximum normal number of the corresponding data type.

      | dst_type | emax |
      | :---: | :---: |
      | `torch_npu.float4_e2m1fn_x2` | 2 |
      | `torch_npu.float4_e1m2fn_x2` | 0 |
      | `torch.float8_e4m3fn` | 8 |
      | `torch.float8_e5m2` | 15 |

  - Scenario 2, when `scale_alg` is 1, applies only to the `torch.float8_e4m3fn` and `torch.float8_e5m2` types:

    - The long vector is divided into blocks, each with length k. A block scaling factor $S_{fp32}^b$ is calculated separately for each block, and all elements in the block are mapped to the target low-precision type by using the same $S_{fp32}^b$. If the last block has fewer than k elements, the missing values are treated as 0 and the block is processed as a complete block.
    - Find the maximum absolute value in the block:
    $$
    Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$
    - When `max_low_bound` is greater than 0, clamp the lower bound of the maximum absolute value:
    $$
    Amax(D_{fp32}^b)=max(Amax(D_{fp32}^b), max\_low\_bound)
    $$
    - Map FP32 to the range representable by the target data type, where $Amax(DType)$ is the maximum value representable by the target precision:
    $$
    S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
    $$
    - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in FP8 format.
    - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scaling factor $S_{fp32}^b$.
    - To prevent overflow during quantization, round the exponent up and keep it within the range representable by FP8:
    $$
    E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, } E_{int}^b < 254, \text{ and } M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a subnormal number and } M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{otherwise} \end{cases}
    $$
    - Calculate the block scaling factor: $S_{ue8m0}^b=2^{E_{int}^b}$
    - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - Apply it to the final quantization step. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantization result is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ is the block scaling factor (that is, $S_{ue8m0}^b$) and $[d^i]_{i=1}^k$ is the quantized data in the block.

  - Scenario 3, when `scale_alg` is 2, applies only to the `torch_npu.float4_e2m1fn_x2` type:

    - When `dst_type_max` is 0.0, 6.0, or 7.0:
      - The input `input` is grouped along the `axis` dimension into groups of k = `block_size` values. A group of k values $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale, \{P_i\}_{i=1}^{k}\}$, k = `block_size`:
      $$
      shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{if the first one or two high bits of the mantissa are 1 and the mantissa is not all 0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{otherwise} \end{cases}
      $$
      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space block\_size
      $$
      - The quantized $P_i$ forms the output `y` at the positions of $V_i$, and `mxscale` forms the output `mxscale` by grouping on the corresponding `axis` dimension.
    - When `dst_type_max` is not 0.0, 6.0, or 7.0:
      - The long vector is divided into blocks, each with length k. A block scaling factor $S_{fp32}^b$ is calculated separately for each block, and all elements in the block are mapped to the target low-precision type by using the same $S_{fp32}^b$. If the last block has fewer than k elements, the missing values are treated as 0 and the block is processed as a complete block.
      - Find the maximum absolute value in the block:
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
      - Map FP32 to the range representable by the target data type. When `dst_type_max` is 0, $Amax(DType)$ is the maximum value representable by the target precision; when `dst_type_max` is not 0, $Amax(DType)$ is the value passed in `dst_type_max`:
      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$
      - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in FP8 format.
      - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scaling factor $S_{fp32}^b$.
      - To prevent overflow during quantization, round the exponent up and keep it within the range representable by FP8:
      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, } E_{int}^b < 254, \text{ and } M_{fixp}^b > 0 \\ E_{int}^b, & \text{otherwise} \end{cases}
      $$
      - Calculate the block scaling factor: $S_{ue8m0}^b=2^{E_{int}^b}$
      - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
      - Apply it to the final quantization step. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantization result is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ is the block scaling factor (that is, $S_{ue8m0}^b$) and $[d^i]_{i=1}^k$ is the quantized data in the block.

## Prototype

```python
torch_npu.npu_dynamic_mx_quant(input, *, axis=-1, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2, block_size=32, scale_alg=0, dst_type_max=0.0, max_low_bound=0.0) -> (Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Input Tensor to be quantized. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The shape can have 1 to 7 dimensions. Non-contiguous Tensors are supported. The data layout can be ND.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`axis`** (`int`): Optional. Axis on which quantization occurs. The value range is [-D, D-1], where D is the number of dimensions of x. The default value is -1.
- **`round_mode`** (`str`): Optional. Data conversion mode. The default value is `"rint"`.
  - When `dst_type` is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, the supported values are `"rint"`, `"floor"`, and `"round"`.
  - When `dst_type` is `torch.float8_e4m3fn` or `torch.float8_e5m2`, only `"rint"` is supported.
- **`dst_type`** (`int`): Optional. Data type of the quantized output y. The supported types are `torch.float8_e5m2`, `torch.float8_e4m3fn`, `torch_npu.float4_e2m1fn_x2`, and `torch_npu.float4_e1m2fn_x2`. The default value is `torch_npu.float4_e2m1fn_x2`.
- **`block_size`** (`int`): Optional. Number of elements quantized each time. It must be a multiple of 32. The value range is (0, 1024]. The default value is 32. When `scale_alg` is 2, only 32 is supported. When the data type of `input` is `torch.float32`, only 32 is supported, and the size of the dimension corresponding to `axis` must not be less than 32.
- **`scale_alg`** (`int`): Optional. Algorithm for calculating `mxscale`. The default value is 0. The value range is 0 (scenario 1), 1 (scenario 2), or 2 (scenario 3).
  - When `dst_type` is `torch_npu.float4_e1m2fn_x2`, only 0 is supported.
  - When `dst_type` is `torch_npu.float4_e2m1fn_x2`, only 0 and 2 are supported.
  - When `dst_type` is `torch.float8_e4m3fn` or `torch.float8_e5m2`, only 0 and 1 are supported.
- **`dst_type_max`** (`float`): Optional. Maximum representable value of the target data type. The value can be 0.0 or 6.0 to 12.0. When the value is 0.0, the maximum value representable by the target precision is used. When the value is 6.0 to 12.0, the passed value is used. The default value is 0.0. This value can be set only when `dst_type` is `torch_npu.float4_e2m1fn_x2` and `block_size` is 32.
- **`max_low_bound`** (`float`): Optional. Lower-bound clamp value of the maximum absolute value calculated for each block. The default value is 0.0, indicating that no clamping is performed. It takes effect only when `scale_alg` is 1. When `scale_alg` is not 1, it must be 0.0. The value is a non-negative number. When the value is greater than 0, the maximum absolute value of each block is compared with `max_low_bound`, and the larger value is used for scale calculation.

## Return Values

- **`y`** (`Tensor`): Quantized output Tensor. When `dst_type` is `torch.float8_e4m3fn` or `torch.float8_e5m2`, the data type of y corresponds to `dst_type`, and the shape is the same as that of the input `input`. When `dst_type` is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, the actual data type of y is `torch.uint8`, and the last dimension of the shape is half the last dimension of `input` (every two FP4 values are packed into one `torch.uint8`). You need to unpack it to view the specific values.
- **`mxscale`** (`Tensor`): Quantization scale corresponding to each group. The logical data type is `torch.float8_e8m0fnu`, and the actual returned data type is `torch.uint8`. The shape of `mxscale` has one more dimension than the input `input`, and the last dimension is 2. The size of the `axis` dimension is the value of the corresponding axis of `input` divided by `block_size`, rounded up and padded to an even number (the pad value is 0). When `axis` is not the last axis, the `mxscale` output must be interleaved for every two rows of data.

## Constraints

- This API supports inference scenarios.
- This API supports single-operator mode and graph mode.
- When `dst_type` is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, the last dimension of `input` must be even.
- Shape constraints between the input `input` and the output `mxscale`:
  - rank(mxscale) = rank(x) + 1
  - axis_change = axis if axis >= 0 else axis + rank(x)
  - mxscale.shape[axis_change] = (ceil(x.shape[axis] / block_size) + 2 - 1) / 2
  - mxscale.shape[-1] = 2
  - Other dimensions are the same as those of the input x.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu

    def dynamic_mx_quant_test(x_dtype, dst_type):
        # Construct the x tensor.
        x = torch.randn((16, 256), dtype=x_dtype).npu()
        y_tmp, mxscale = torch_npu.npu_dynamic_mx_quant(x, axis=-1, round_mode="rint", dst_type=dst_type, block_size=32, scale_alg=0)
        y = y_tmp.cpu()
        mxscale = mxscale.cpu()
        print("DynamicMxQuant result:")
        print("x:\n", x)
        print("y:\n", y)
        print("mxscale:\n", mxscale)
    if __name__ == "__main__":
        dynamic_mx_quant_test(torch.bfloat16, torch.float8_e4m3fn)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair

    class DynamicMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, axis=-1, round_mode='rint', dst_type=torch_npu.float8_e5m2, block_size=32, scale_alg=0):
            return torch_npu.npu_dynamic_mx_quant(x, axis=axis, round_mode=round_mode, dst_type=dst_type, block_size=block_size, scale_alg=scale_alg)

    def dynamic_mx_quant_test(x_dtype, dst_type):
        # Construct the x tensor.
        x = torch.randn((16, 256), dtype=x_dtype).npu()
        model = DynamicMxQuantModel()
        model.to('npu')
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        y_tmp, mxscale = model(x, axis=-1, dst_type=dst_type, block_size=32, scale_alg=0)
        y = y_tmp.cpu()
        mxscale = mxscale.cpu()
        print("DynamicMxQuant result:")
        print("x:\n", x)
        print("y:\n", y)
        print("mxscale:\n", mxscale)
    if __name__ == "__main__":
        dynamic_mx_quant_test(torch.float16, torch.float8_e5m2)
    ```
