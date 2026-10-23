# torch_npu.npu_dynamic_block_mx_quant

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- API function: Performs MX quantization on the input variable by using data blocks (32*32) as basic blocks and converts the result to the target data type.

    In each basic block, a different scale algorithm is used according to the value of `scale_alg` to calculate the quantization parameter `scale` (1*1) of the current block. It is broadcast to `scale1` (32*1) and `scale2` (1*32) for output. At the same time, each number in the basic block is divided by `scale` and converted to the corresponding `dst_type` according to `round_mode` to obtain the quantization result `y`.

- Formulas:
  - Scenario 1, when `scale_alg` is 0:
    - The input is grouped by data blocks (32*32) as basic blocks. The numbers in a data block $\{\{V_i\}_{i=1}^{32*32}\}$ are quantized into $\{scale, \{P_i\}_{i=1}^{32*32}\}$.

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
      scale = 2^{shared\_exp} \\
      P_i = cast\_to\_dst\_type(V_i/scale, round\_mode), \space i\space from\space 1\space to\space 32*32
      $$

    - At the same time, scale (1*1) is broadcast to scale1 (32*1) and scale2 (1*32) as the output scale1 and scale2. The quantized $P_i$ is placed at the corresponding $V_i$ position to form the output y.

    - emax: Exponent bit of the maximum normal number of the corresponding data type.

      | dst_type | emax |
      | :---: | :---: |
      | `torch_npu.float4_e2m1fn_x2` | 2 |
      | `torch_npu.float4_e1m2fn_x2` | 0 |
      | `torch.float8_e4m3fn` | 8 |
      | `torch.float8_e5m2` | 15 |

  - Scenario 2, when `scale_alg` is 2, only the `torch_npu.float4_e2m1fn_x2` type is involved:
    - The input is divided into blocks, each with length k. A block scaling factor $S_{fp32}^b$ is calculated separately for each block, and all elements in the block are mapped to the target low-precision type `torch_npu.float4_e2m1fn_x2` by using the same $S_{fp32}^b$. The scale storage type is `torch.float8_e8m0fnu`. If the last block has fewer than k elements, the missing values are treated as 0 and the block is processed as a complete block.
    - Find the maximum absolute value in the block:

      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$

    - When `dst_type_max` is not 0, the scale is calculated by using the passed value. When `dst_type_max` is 0, the maximum value of the target data type is used.
    - Map FP32 to the range representable by the target data type:

      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{dst\_type\_max}
      $$

    - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in FP8 format.
    - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scaling factor $S_{fp32}^b$.
    - To prevent overflow during quantization, round the exponent up and keep it within the range representable by FP8:

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, } E_{int}^b < 254, \text{ and } M_{fixp}^b > 0 \\ E_{int}^b, & \text{otherwise} \end{cases}
      $$

    - Calculate the block scaling factor: $S_{ue8m0}^b=2^{E_{int}^b}$.
    - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$.
    - Apply it to the final quantization step. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantization result is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ represents the block scaling factor, that is, $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ represents the quantized data in the block.

## Prototype

```python
torch_npu.npu_dynamic_block_mx_quant(input, *, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2, scale_alg=0, dst_type_max=0.0) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Data to be quantized. The data type can be `torch.float16` or `torch.bfloat16`. The shape can have 2 or 3 dimensions. Non-contiguous Tensors are supported. The data layout can be ND. When `dst_type` is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, the last dimension of `input` must be even. Empty Tensors are not supported.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`round_mode`** (`str`): Optional. Mode for casting the quantization result to the data type of the output y. The default value is `"rint"`.
  - When `dst_type` is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, the supported values are `"rint"`, `"floor"`, and `"round"`.
  - When `dst_type` is `torch.float8_e5m2` or `torch.float8_e4m3fn`, only `"rint"` is supported.
- **`dst_type`** (`int`): Optional. Data type of the output y. The supported types are `torch.float8_e5m2`, `torch.float8_e4m3fn`, `torch_npu.float4_e2m1fn_x2`, and `torch_npu.float4_e1m2fn_x2`. The default type is `torch_npu.float4_e2m1fn_x2`.
- **`scale_alg`** (`int`): Optional. Scale calculation method. The default value is 0. The value range is 0 (scenario 1) or 2 (scenario 2). When `dst_type` is `torch_npu.float4_e1m2fn_x2`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`, only 0 is supported.
- **`dst_type_max`** (`float`): Optional. Maximum value of the target data type. The default value is 0.0. This value can be set only when `scale_alg`=2 and `dst_type` is `torch_npu.float4_e2m1fn_x2`. Currently, only 0.0, 6.0, and 7.0 are supported.

## Return Values

- **`y`** (`Tensor`): Result after quantizing `input`. The data type is specified by the `dst_type` parameter, and the shape is the same as that of the input `input`. When `dst_type` is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, the actual data type of y is `torch.uint8`, and you need to unpack it to view the specific values.
- **`scale1_out`** (`Tensor`): Quantization scale corresponding to each group on the last axis. The logical data type is `torch.float8_e8m0fnu`, and the actual data type is `torch.uint8`. You need to convert it to view the specific values. The shape is the value of the last axis of `input` divided by 32, rounded up, and padded to an even number with 0.
- **`scale2_out`** (`Tensor`): Quantization scale corresponding to each group on the second-to-last axis. The logical data type is `torch.float8_e8m0fnu`, and the actual data type is `torch.uint8`. You need to convert it to view the specific values. The shape is the value of the second-to-last axis of `input` divided by 32, rounded up, and padded to an even number with 0. The `scale2_out` output must be interleaved for every two rows of data.

## Constraints

- This API supports inference scenarios.
- This API supports single-operator mode and graph mode.
- When the output target type is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, the last dimension of `input` must be even.
- Shape constraints between the input `input` and the outputs `scale1_out` and `scale2_out`:
  - rank(scale1_out) = rank(input) + 1
  - rank(scale2_out) = rank(input) + 1
  - scale1_out.shape[-2] = (ceil(input.shape[-1] / 32) + 2 - 1) / 2
  - scale2_out.shape[-3] = (ceil(input.shape[-2] / 32) + 2 - 1) / 2
  - scale1_out.shape[-1] = 2
  - scale2_out.shape[-1] = 2
  - Other dimensions are the same as those of the input `input`.
  - For example, if the shape of the input `input` is [B, M, N], the shape of y is [B, M, N], the shape of scale1_out is [B, M, (ceil(N/32)+2-1)/2, 2], and the shape of scale2_out is [B, (ceil(M/32)+2-1)/2, N, 2].

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu

    def dynamic_block_mx_quant_test(x_dtype, round_mode, dst_type, scale_alg, dst_type_max):
        # Construct the x tensor.
        x = torch.randn((1, 4), dtype=x_dtype).npu()
        y_tmp, scale1_tmp, scale2_tmp = torch_npu.npu_dynamic_block_mx_quant(x, round_mode=round_mode,
        dst_type=dst_type,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max)
        y = y_tmp.cpu()
        scale1 = scale1_tmp.cpu()
        scale2 = scale2_tmp.cpu()
        print("DynamicBlockMxQuant result:")
        print("x:\n", x)
        print("y:\n", y)
        print("scale1:\n", scale1)
        print("scale2:\n", scale2)

    if __name__ == "__main__":
        dynamic_block_mx_quant_test(torch.bfloat16, 'rint', torch_npu.float4_e2m1fn_x2, 0, 0.0)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair

    class DynamicBlockMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, round_mode='rint', dst_type=torch_npu.float4_e1m2fn_x2, scale_alg=0, dst_type_max=0.0):
            return torch_npu.npu_dynamic_block_mx_quant(x, round_mode=round_mode, dst_type=dst_type, scale_alg=scale_alg, dst_type_max=dst_type_max)

    def dynamic_block_mx_quant_test(x_dtype, round_mode, dst_type, scale_alg, dst_type_max):
        # Construct the x tensor.
        x = torch.randn((1, 4), dtype=x_dtype).npu()
        model = DynamicBlockMxQuantModel()
        model.to('npu')
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        y_tmp, scale1_tmp, scale2_tmp = model(x, round_mode=round_mode, dst_type=dst_type, scale_alg=scale_alg, dst_type_max=dst_type_max)
        y = y_tmp.cpu()
        scale1 = scale1_tmp.cpu()
        scale2 = scale2_tmp.cpu()
        print("DynamicBlockMxQuant result:")
        print("x:\n", x)
        print("y:\n", y)
        print("scale1:\n", scale1)
        print("scale2:\n", scale2)

    if __name__ == "__main__":
        dynamic_block_mx_quant_test(torch.bfloat16, "rint", torch_npu.float4_e2m1fn_x2, 0, 0.0)
    ```
