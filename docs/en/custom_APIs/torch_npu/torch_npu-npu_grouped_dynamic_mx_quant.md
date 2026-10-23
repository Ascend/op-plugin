# torch_npu.npu_grouped_dynamic_mx_quant

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->

## Function

- API function: Groups the input x according to the starting values of the passed group indices (`group_index`), and performs dynamic MX quantization with the target data type `float8` (`torch.float8_e4m3fn`, `torch.float8_e5m2`) or `float4` (`torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2`) on the data at the granularity of basic blocks (`blocksize`). It outputs the quantization scale `mxscale` (`torch.float8_e8m0fnu`).

- Formulas:
  - Scenario 1, when scale_alg is 0:
    - The input x is first grouped on dimension 0 according to group_index. Within each group, values are grouped by k = blocksize. For a group of k values $\{\{V_i\}_{i=1}^{k}\}$, the quantization scale mxscale_pre corresponding to the group is calculated, yielding $\{mxscale\_pre, \{P_i\}_{i=1}^{k}\}$:

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale\_pre = 2^{shared\_exp}
    $$

    - Each value in the group is divided by mxscale_pre and converted to the corresponding dst_type according to round_mode to obtain the quantization result y:

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale\_pre, round\_mode), \space i\space from\space 1\space to\space blocksize
    $$

    - The quantized $P_i$ forms the output y at the positions of $V_i$. mxscale_pre is grouped according to the corresponding group_index, and the first dimension in the group is padded to an even number to form the output mxscale.
    - emax: Exponent bit of the maximum normal number of the corresponding data type. The mapping is as follows.

      | dst_type | emax |
      | --- | --- |
      | `torch_npu.float4_e2m1fn_x2` | 2 |
      | `torch_npu.float4_e1m2fn_x2` | 0 |
      | `torch.float8_e4m3fn` | 8 |
      | `torch.float8_e5m2` | 15 |

  - Scenario 2, when scale_alg is 1 (only float8 is involved):
    - The long vector is divided into blocks, each with length k. A block scaling factor $S_{fp32}^b$ is calculated separately for each block, and all elements in the block are mapped to the target low-precision type float8 by using the same $S_{fp32}^b$. If the last block has fewer than k elements, the missing values are treated as 0 and the block is processed as a complete block.
    - Find the maximum absolute value in the block:

    $$
    Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$

    - Map fp32 to the range representable by the target data type float8, where $Amax(DType)$ is the maximum value representable by the target precision:

    $$
    S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
    $$

    - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in fp8 format. Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from $S_{fp32}^b$. To prevent overflow during quantization, round the exponent up and keep it within the range representable by fp8:

    $$
    E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, } E_{int}^b < 254, \text{ and } M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a subnormal number and } M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{otherwise} \end{cases}
    $$

    - Calculate the block scaling factor $S_{ue8m0}^b=2^{E_{int}^b}$ and the block conversion factor $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The final quantization result is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ is the block scaling factor (that is, $S_{ue8m0}^b$) and $[d^i]_{i=1}^k$ is the quantized data in the block.

  - Scenario 3, when scale_alg is 2 (only float4 types are involved):
    - When dst_type_max is 0.0/6.0/7.0 (`torch_npu.float4_e2m1fn_x2`) or 1.875 (`torch_npu.float4_e1m2fn_x2`):
      - The input x is grouped on dimension 0 by k = blocksize values. A group of k values $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale, \{P_i\}_{i=1}^{k}\}$, k = blocksize:

      $$
      shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{if the first 1/2/3 high bits of the mantissa are 1 and the mantissa is not all 0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{otherwise} \end{cases}
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize
      $$

      - The quantized $P_i$ forms the output y at the positions of $V_i$, and the output mxscale is formed by grouping on the corresponding dimension.
    - When dst_type_max is not one of the preceding special values (for `torch_npu.float4_e2m1fn_x2`, other values in 6.0-12.0; for `torch_npu.float4_e1m2fn_x2`, other values in 1.75-3.5):
      - The long vector is divided into blocks, each with length k. A block scaling factor $S_{fp32}^b$ is calculated separately for each block, and all elements in the block are mapped to the target low-precision type by using the same $S_{fp32}^b$. If the last block has fewer than k elements, the missing values are treated as 0 and the block is processed as a complete block.
      - Find the maximum absolute value in the block:

      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$

      - Map fp32 to the range representable by the target data type. When dst_type_max is 0, $Amax(DType)$ is the maximum value representable by the target precision. When dst_type_max is not 0, $Amax(DType)$ is the value passed in dst_type_max:

      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$

      - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in fp8 format. Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from $S_{fp32}^b$. To prevent overflow during quantization, round the exponent up and keep it within the range representable by fp8:

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, } E_{int}^b < 254, \text{ and } M_{fixp}^b > 0 \\ E_{int}^b, & \text{otherwise} \end{cases}
      $$

      - Calculate the block scaling factor $S_{ue8m0}^b=2^{E_{int}^b}$ and the block conversion factor $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantization result is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ is the block scaling factor (that is, $S_{ue8m0}^b$) and $[d^i]_{i=1}^k$ is the quantized data in the block.

## Prototype

```python
torch_npu.npu_grouped_dynamic_mx_quant(x, group_index, *, round_mode="rint", dst_type=23, blocksize=32,
                                       scale_alg=0, dst_type_max=0.0) -> (Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input Tensor, $x$ in the formulas. Only 2 dimensions are supported. The data layout can be ND. The data type can be `torch.float16` or `torch.bfloat16`. Non-contiguous Tensors and empty Tensors are supported.
- **`group_index`** (`Tensor`): Required. Starting index of the quantization group. Only 1 dimension is supported. The data layout can be ND. The data type can be `torch.int32`. Non-contiguous Tensors are supported, and empty Tensors are not supported. The indices must be greater than or equal to 0 and non-decreasing, and the last value must be equal to the size of dimension 0 of x.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`round_mode`** (`str`): Optional. Data conversion mode, $round\_mode$ in the formulas. The default value is "rint".
  - When dst_type is `torch.float8_e5m2` or `torch.float8_e4m3fn`, only "rint" is supported.
  - When dst_type is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, "rint", "round", and "floor" are supported.
- **`dst_type`** (`int`): Optional. Data type of y after conversion. The default value is 23. The supported values are:
  - 23: `torch.float8_e5m2`.
  - 24: `torch.float8_e4m3fn`.
  - 296: `torch_npu.float4_e2m1fn_x2`.
  - 297: `torch_npu.float4_e1m2fn_x2`.
- **`blocksize`** (`int`): Optional. Number of elements quantized each time, $blocksize$ in the formulas. Currently, only 32 is supported. The default value is 32.
- **`scale_alg`** (`int`): Optional. Algorithm used for mxscale calculation. The value can be 0, 1, or 2. The default value is 0.
  - 0: Scenario 1 (shared-exponent MX quantization). Both float8 and float4 are supported.
  - 1: Scenario 2 (per-block scaling). Only float8 is supported.
  - 2: Scenario 3 (fp4 dynamic quantization). Only float4 is supported.
- **`dst_type_max`** (`float`): Optional. Value of maxType, $Amax(DType)$ in the formulas. It takes effect only when dst_type is float4. The default value is 0.0.
  - 0.0: Amax(DType) is the maximum value of the quantized result data type.
  - `torch_npu.float4_e2m1fn_x2`: 0.0 and 6.0-12.0 are supported.
  - `torch_npu.float4_e1m2fn_x2`: 0.0 and 1.75-3.5 are supported.

## Return Values

- **`y`** (`Tensor`): Quantized output Tensor, $y$ in the formulas. The shape is the same as that of the input x. When dst_type is float8, the data type is `torch.float8_e5m2` or `torch.float8_e4m3fn`. When dst_type is float4, the actual returned data type is `torch.uint8`, and the last dimension is half the last dimension of the input x (each `torch.uint8` element packs two float4 values). You need to unpack it to view the specific values, such as `y.view(torch_npu.float4_e2m1fn_x2)`. Empty Tensors are supported. Non-contiguous Tensors are supported for float8 but not for float4.
- **`mxscale`** (`Tensor`): Quantization scale corresponding to each group, $mxscale$ in the formulas. The logical data type is `torch.float8_e8m0fnu`, and the actual returned data type is `torch.uint8`. You need to convert it to view the specific values, such as `mxscale.view(torch.float8_e8m0fnu)`. If the shape of x is [m, n] and the shape of `group_index` is [g], the shape of `mxscale` is [(m/(blocksize∗2)+g), n, 2]. Empty Tensors are supported, and non-contiguous Tensors are not supported.

## Constraints

- This API supports training and inference scenarios.
- This API supports single-operator mode and graph mode.
- Deterministic computation: The deterministic implementation is used by default.
- Shape constraints for x, group_index, y, and mxscale are as follows (assuming that the shape of x is [m, n] and the shape of group_index is [g]):
  - rank(mxscale) = rank(x) + 1.
  - mxscale.shape[0] = m / (blocksize ∗ 2) + g.
  - mxscale.shape[-1] = 2. Other dimensions are the same as those of the input x.
  - The shape of the output y is the same as that of x (for float4 output, the actual last dimension of y is half the last dimension of x).
- The parameter combination constraints for dst_type, scale_alg, round_mode, and dst_type_max are as follows.

| dst_type | scale_alg | round_mode | dst_type_max |
| --- | --- | --- | --- |
| `torch.float8_e5m2` | 0, 1 | "rint" | Cannot be set (only the default value 0.0) |
| `torch.float8_e4m3fn` | 0, 1 | "rint" | Cannot be set (only the default value 0.0) |
| `torch_npu.float4_e2m1fn_x2` | 0, 2 | "rint"/"round"/"floor" | 0.0 and 6.0-12.0 |
| `torch_npu.float4_e1m2fn_x2` | 0, 2 | "rint"/"round"/"floor" | 0.0 and 1.75-3.5 |

## Examples

- Single-operator call, float8 quantization (`scale_alg` uses the default value 0)

    ```python
    import torch
    import torch_npu
    
    # The shape of x is [8, 1], and the values are (0, 8, 64, 512) repeated twice. group_index=[4, 8] indicates two groups: [0:4] and [4:8].
    x = torch.tensor([[0], [8], [64], [512], [0], [8], [64], [512]], dtype=torch.bfloat16).npu()
    group_index = torch.tensor([4, 8], dtype=torch.int32).npu()
    y, mxscale = torch_npu.npu_grouped_dynamic_mx_quant(x, group_index, dst_type=torch.float8_e4m3fn)
    print(y)
    print(mxscale)
    print(mxscale.cpu().view(torch.float8_e8m0fnu))
    ```

    The output is as follows:

    ```text
    tensor([[  0.],
            [  4.],
            [ 32.],
            [256.],
            [  0.],
            [  4.],
            [ 32.],
            [256.]], device='npu:0', dtype=torch.float8_e4m3fn)
    tensor([[[128,   0]],
    
            [[128,   0]]], device='npu:0', dtype=torch.uint8)
    tensor([[[2., 0.]],
    
            [[2., 0.]]], dtype=torch.float8_e8m0fnu)
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

        def forward(self, x, group_index):
            return torch_npu.npu_grouped_dynamic_mx_quant(x, group_index, dst_type=torch.float8_e4m3fn)

    model = Model().npu()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    x = torch.randn((64, 256), dtype=torch.float16).npu()
    group_index = torch.tensor([32, 64], dtype=torch.int32).npu()
    y, mxscale = model(x, group_index)
    print(y)
    print(mxscale)
    ```

    The output is as follows:

    ```text
    tensor([[-128.0000,  208.0000, -160.0000,  ...,  -32.0000, -112.0000,
              128.0000],
            [-144.0000,  104.0000, -208.0000,  ...,  112.0000,  288.0000,
              120.0000],
            [  -1.7500,  -80.0000,  240.0000,  ..., -192.0000, -160.0000,
               48.0000],
            ...,
            [-160.0000,   40.0000, -160.0000,  ...,  160.0000,  288.0000,
             -384.0000],
            [-144.0000, -352.0000,    3.5000,  ...,  176.0000,  128.0000,
             -384.0000],
            [ -20.0000,   22.0000,  320.0000,  ...,  256.0000,   -4.5000,
              -96.0000]], device='npu:0', dtype=torch.float8_e4m3fn)
    tensor([[[120,   0],
             [120,   0],
             [119,   0],
             ...,
             [120,   0],
             [120,   0],
             [120,   0]],
    
            [[120,   0],
             [120,   0],
             [119,   0],
             ...,
             [120,   0],
             [120,   0],
             [119,   0]],
    
            [[  0,   0],
             [  0,   0],
             [  0,   0],
             ...,
             [  0,   0],
             [  0,   0],
             [  0,   0]]], device='npu:0', dtype=torch.uint8)
    ```
