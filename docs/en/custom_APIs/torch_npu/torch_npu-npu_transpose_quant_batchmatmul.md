# torch\_npu.npu\_transpose\_quant\_batchmatmul

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## Function

- Performs quantized matrix multiplication on tensor <code>x1</code> and tensor <code>x2</code>. Only three-dimensional tensors are supported. Tensors can be transposed based on the input index arrays. <code>perm_x1</code> and <code>perm_x2</code> represent the permutation sequences for tensors <code>x1</code> and <code>x2</code>, respectively. The dimension with a sequence value of <code>0</code> represents the batch dimension, while the remaining two dimensions are used for matrix multiplication.

- Formula:

    T1, T2, and Ty are described by the parameters `perm_x1`, `perm_x2`, and `perm_y`, respectively, which specify the transpose sequences.

    $$
    out=((x1^{T1}@x2^{T2}+bias)*x2\_scale*x1\_scale)^{Ty}
    $$

## Prototype

```python
torch_npu.npu_transpose_quant_batchmatmul(x1, x2, dtype, *, bias=None, x1_scale=None, x2_scale=None, group_sizes=None, perm_x1=None, perm_x2=None, perm_y=None, batch_split_factor=1, x1_dtype=None, x2_dtype=None) -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. First matrix in the matrix multiplication. The data layout can be `ND`. Only 3D input is supported, with shape `(m, b, k)`. The data type can be `float8_e5m2`, `float8_e4m3fn`, or `hifloat8`.
- **`x2`** (`Tensor`): Required. Second matrix in the matrix multiplication. The data layout can be `ND` or `NZ`. Only 3D input is supported, with shape `(b, k, n)` or `(b, n, k)`. The size of the `k` dimension of `x2` must be the same as that of `x1`. The data type can be `float8_e5m2`, `float8_e4m3fn`, or `hifloat8`.
- **`dtype`** (`int`): Required. Data type of the output. Supported values are `torch.float16`, `torch.bfloat16`, and `torch_npu.hifloat8`.
- **`*`**: Position delimiter. Variables before this delimiter are position-dependent and must be passed in order. Variables after this delimiter are optional keyword arguments and must be assigned using key-value pairs. If not specified, their default values are used.
- **`bias`** (`Tensor`): Optional. Bias matrix for the matrix multiplication. This parameter is not supported in the current version. Use the default value.
- **`x1_scale`** (`Tensor`): Optional. Quantization scale for the left matrix. The data layout can be `ND`. The data type can be `float32`, `float8_e8m0fnu`, or `int64`. The shape can be 1D or 4D.
- **`x2_scale`** (`Tensor`): Optional. Quantization scale for the right matrix. The data layout can be `ND`. The data type can be `float32`, `float8_e8m0fnu`, or `int64`. The shape can be 1D or 4D.
- **`group_sizes`** (`List[int]`): Optional. Quantization group sizes. The data type is `int32`. The default value is `None`.
  - Only a 3-element list is supported, in the form `[group_m, group_n, group_k]`, which specifies the quantization group sizes along the `m`, `n`, and `k` dimensions, respectively. For example, `group_m` specifies that every `group_m` elements along the `m` dimension correspond to one quantization parameter.
  - When one or more values in `[group_m, group_n, group_k]` are `0`, the API sets those values based on the input shapes of `x1`, `x2`, `x1_scale`, and `x2_scale`. For example, when `group_m = 0`, the quantization group size along the `m` dimension is inferred by the API according to the formula `group_m = m / scale_m`, where `m` is divisible by `scale_m`. Here, `m` refers to the `m` dimension of `x1`, and `scale_m` refers to the `m` dimension of `x1_scale`.
  - This parameter is required only in MX quantization mode. Currently, `[group_m, group_n, group_k]` supports only `[0,0,32]`, `[0,1,32]`, `[1,0,32]`, and `[1,1,32]`.

- **`perm_x1`** (`List[int]`): Optional. Transpose sequence for the first matrix in the matrix multiplication. The size is `3`, and the data type is `int64`. The data layout can be `ND`. Only `[1, 0, 2]` is supported.
- **`perm_x2`** (`List[int]`): Optional. Transpose sequence for the second matrix in the matrix multiplication. The size is `3`, and the data type is `int64`. The data layout can be `ND`. `[0, 1, 2]` and `[0, 2, 1]` are supported.
- **`perm_y`** (`List[int]`): Optional. Transpose sequence for the output matrix of the matrix multiplication. The size is `3`, and the data type is `int64`. The data layout can be `ND`. Only `[1, 0, 2]` is supported.
- **`batch_split_factor`** (`int`): Optional. Specifies the split size of the `b` dimension of the output matrix of the matrix multiplication. The data type is `int32`. The default value is `1`. Currently, only `1` is supported.
- **`x1_dtype`** (`int`): Optional. Data type of `x1`. Supported values are `torch.float8_e5m2`, `torch.float8_e4m3fn`, and `torch_npu.hifloat8`.
- **`x2_dtype`** (`int`): Optional. Data type of `x2`. Supported values are `torch.float8_e5m2`, `torch.float8_e4m3fn`, and `torch_npu.hifloat8`.

## Return Values

**`y`** (`Tensor`): Final computation result, `$out$` in the formula. The data layout can be `ND`. Only 3D output is supported, with shape `(m, b, n)`. The data type can be `float16`, `bfloat16`, or `torch_npu.hifloat8`.

## Constraints

- This API can be used in training and inference scenarios.
- This API supports only single-operator mode calls.
- In K-C quantization scenarios:
  - `x1_scale` and `x2_scale` support only 1D input. `x1_scale` must have shape `(m,)`, and `x2_scale` must have shape `(n,)`.
  - `x2` supports only `ND` format.
  - `k` supports only `512`, and `n` supports only `128`.
  - `perm_x2` supports only `[0, 1, 2]`.

- In MX quantization scenarios:
  - `x1` and `x2` support only `float8_e4m3fn` input. `k` must be a multiple of `64`.
  - `x1_scale` and `x2_scale` support only 4D input. `x1_scale` must have shape `(m, b, k/64, 2)`. When `perm_x2` is `[0, 1, 2]`, `x2_scale` must have shape `(b, k/64, n, 2)`; when `perm_x2` is `[0, 2, 1]`, `x2_scale` must have shape `(b, n, k/64, 2)`.

- In T-C quantization scenarios:
  - `x1` and `x2` support only `torch_npu.hifloat8` input.
  - `x2` supports only `ND` format.
  - `x1_scale` and `x2_scale` support only 1D input. `x1_scale` can be empty; when non-empty, it must have shape `(1,)`. `x2_scale` must have shape `(n,)`.

## Example

Single mode call

```python
import torch
import torch_npu

M, K, N, Batch = 32, 512, 128, 32
x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()

x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32).npu()
x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32).npu()
y = torch_npu.npu_transpose_quant_batchmatmul(x1, x2, dtype=torch.float16, x1_scale=x1_scale,
                                        x2_scale=x2_scale, perm_x1=[1, 0, 2],
                                        perm_x2=[0, 1, 2], perm_y=[1, 0, 2])
```
