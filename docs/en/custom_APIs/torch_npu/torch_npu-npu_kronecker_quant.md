# torch_npu.npu_kronecker_quant

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- **API function**: In inference scenarios, this API performs two matrix multiplications on the input Tensor in sequence and quantizes the matrix multiplication result to the `torch_npu.int4` type or `torch_npu.float4_e2m1fn_x2` type.

  pertoken quantization supports the `torch_npu.int4` output type and packs every eight values into the `torch.int32` type for output. It also outputs a `torch.float32` quantization scaling coefficient.

  pergroup quantization supports outputting the `torch_npu.float4_e2m1fn_x2` quantization result and the `torch.float8_e8m0fnu` quantization coefficient. Elements in the group share one quantization parameter.

- **pertoken formulas**:

    1. Right-multiply the input `x` by `kroneckerP2`:

        $$
        x' = x @ kroneckerP2
        $$

    2. Left-multiply `x'` by `kroneckerP1`:

        $$
        x'' = kroneckerP1 @ x'
        $$

    3. Compute the maximum absolute value along dimension 0 of $x''$ and divide it by $(7 / clipRatio)$ to calculate the quantization scaling coefficient `quantScale` to be quantized to the `torch_npu.int4` format:

        $$
        quantScale = \frac{[\max(\operatorname{abs}(x''[0, :, :])),\ \max(\operatorname{abs}(x''[1, :, :])),\ \ldots,\ \max(\operatorname{abs}(x''[K, :, :]))]}{7 / clipRatio}
        $$

    4. Compute the output `out`:

        $$
        out = x'' / quantScale
        $$

- **pergroup formulas**:

    1. Right-multiply the input `x` by `kroneckerP2`:

        $$
        x' = x @ kroneckerP2
        $$

    2. Left-multiply `x'` by `kroneckerP1`:

        $$
        x'' = kroneckerP1 @ x'
        $$

    3. To perform pergroup quantization on $x''$, the shape must be converted and is recorded as `x2`. For example, [K,M,N] is converted to [K,M*N]. Per-group quantization is performed along the second dimension of $x''$. A group contains elements with exponents $e_0, e_1, \ldots, e_{31}$. Calculate $emax$:

        $$
        emax = \max(e_0, e_1, \ldots, e_{31})
        $$

    4. Calculate reduceMaxValue and sharedExp:

        $$
        reduceMaxValue = \log_2(\operatorname{reduceMax}(x2)),\ groupsize = 32
        $$

        $$
        sharedExp = reduceMaxValue - emax
        $$

    5. Calculate quantScale:

        $$
        quantScale = 2^{sharedExp}
        $$

    6. Each blocksize shares one quantscale, and out is calculated:

        $$
        out = x2 / quantScale
        $$

## Prototype

```python
torch_npu.npu_kronecker_quant(x, kronecker_p1, kronecker_p2, clip_ratio=None, dst_dtype=None, dst_type_max=None) -> (Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Source data Tensor to be quantized, `x` in the formulas. The data type can be `torch.float16` or `torch.bfloat16`. The shape is (K, M, N). The data layout can be ND.
- **`kronecker_p1`** (`Tensor`): Required. Left multiplication matrix of `x`, `kroneckerP1` in the formulas. The data type is the same as that of `x`. The shape is (M, M). The data layout can be ND.
- **`kronecker_p2`** (`Tensor`): Required. Right multiplication matrix of `x`, `kroneckerP2` in the formulas. The data type is the same as that of `x`. The shape is (N, N) (when `dst_dtype` is `torch_npu.float4_e2m1fn_x2`, the shape can be (0,0)). The data layout can be ND.
- **`clip_ratio`** (`float`): Optional. Clipping ratio for quantization, `clipRatio` in the formulas. The data type can be `float`. The value range is (0, 1.0]. The default value is None, indicating clipping at a ratio of 1.0.
- **`dst_dtype`** (`ScalarType`): Optional. Target output type of quantization. If the output type is `torch_npu.float4_e2m1fn_x2`, this value is `torch_npu.float4_e2m1fn_x2`. If the output type is `torch.int`, this parameter does not need to be filled.
- **`dst_type_max`** (`float`): Optional. Maximum value of the quantized data target. The data type can be `float`. The value range is 0.0 or 6.0-12.0. The value 0.0 indicates that this parameter is not used. The value 6.0-12.0 indicates the maximum value of the target data type. This value can be set only when the data type is `torch_npu.float4_e2m1fn_x2`.

## Return Values

- **`out`** (`Tensor`): Quantized output, `out` in the formulas. The data layout can be ND.
  - pertoken quantization: The data type can be `torch.int32`, and the shape is (K, M, N / 8).
  - pergroup quantization: The data type can be `torch_npu.float4_e2m1fn_x2`, and the shape is (K, M*N).

- **`quant_scale`** (`Tensor`): Quantization scaling coefficient, `quantScale` in the formulas. The data layout can be ND.
  - pertoken quantization: The data type can be `torch.float32`, and the shape is (K,).
  - pergroup quantization: The data type can be `torch.float8_e8m0fnu`, and the shape is (K, ceilDiv(M*N, 64), 2). For example, if the input shape is (16, 128, 64), the shape of `out` is (16, 8192) and the shape of `quant_scale` is (16, 128, 2).

## Constraints

- This API supports inference scenarios only.
- This API supports only single-operator mode and TorchAir graph mode.
- Variables used in the shape description:
  - K: Size of the first dimension of the input Tensor `x`. The value range is [1, 262144].
  - M: Size of the middle dimension of the input Tensor `x`. The value range is [1, 256].
  - N: Size of the last dimension of the input Tensor `x`. The value range is [1, 256], and N must be even. If the output type of `out` is `torch.int32`, N must be a multiple of 8.

## Examples

- Single-operator call

  - Output type `torch_npu.int4`

    ```python
    import torch
    import torch_npu

    K = 16
    M = 64
    N = 64
    x = torch.randn(K, M, N).half().npu()
    p1 = torch.randn(M, M).half().npu()
    p2 = torch.randn(N, N).half().npu()
    clip_ratio = 1.0
    out, quant_scale = torch_npu.npu_kronecker_quant(x, p1, p2, clip_ratio)
    print(out)
    print(quant_scale)
    ```

  - Output type `torch_npu.float4_e2m1fn_x2`

    ```python
    import torch
    import torch_npu
    K = 16
    M = 64
    N = 64
    x = torch.randn(K, M, N).half().npu()
    p1 = torch.randn(M, M).half().npu()
    p2 = torch.randn(N, N).half().npu()
    clip_ratio = 1.0
    dst_type_max = 0.0
    dst_dtype = torch_npu.float4_e2m1fn_x2
    out, quant_scale = torch_npu.npu_kronecker_quant(x, p1, p2, clip_ratio, dst_dtype, dst_type_max)
    print(out.cpu())
    print(quant_scale.cpu())
    ```

- Graph mode call

  - Output type `torch_npu.int4`

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, p1, p2, clip_ratio):
                return torch_npu.npu_kronecker_quant(x, p1, p2, clip_ratio)

    K = 16
    M = 64
    N = 64
    x = torch.randn(K, M, N).half().npu()
    p1 = torch.randn(M, M).half().npu()
    p2 = torch.randn(N, N).half().npu()
    clip_ratio = 1.0

    module = torch.compile(Module().npu(), backend=npu_backend)
    out, quant_scale = module(x, p1, p2, clip_ratio)
    print(out)
    print(quant_scale)
    ```

  - Output type `torch_npu.float4_e2m1fn_x2`

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, p1, p2, clip_ratio):
                return torch_npu.npu_kronecker_quant(x, p1, p2, clip_ratio, torch_npu.float4_e2m1fn_x2, dst_type_max)

    K = 16
    M = 64
    N = 64
    x = torch.randn(K, M, N).half().npu()
    p1 = torch.randn(M, M).half().npu()
    p2 = torch.randn(N, N).half().npu()
    clip_ratio = 1.0
    dst_type_max = 0.0

    module = torch.compile(Module().npu(), backend=npu_backend)
    out, quant_scale = module(x, p1, p2, clip_ratio)
    print(out.cpu())
    print(quant_scale.cpu())
    ```
