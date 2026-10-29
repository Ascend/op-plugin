# torch\_npu.npu\_transpose\_quant\_batchmatmul

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- Function: performs the quantized matrix multiplication of the tensor x1 and the tensor x2. Only 3D tensors are supported. The tensors support transposition, and the transposition sequence changes according to the passed sequence. perm\_x1 indicates the transposition sequence of the tensor x1, and perm\_x2 indicates the transposition sequence of the tensor x2. The dimension with the sequence value 0 is the batch dimension, and the matrix multiplication is performed on the other two dimensions.
- Formulas:

    T1, T2, and Ty describe the transposition sequences through the parameters perm\_x1, perm\_x2, and perm\_y respectively.

    $$
    out=((x1^{T1}@x2^{T2}+bias)*x2\_scale*x1\_scale)^{Ty}
    $$

## Prototype

```python
torch_npu.npu_transpose_quant_batchmatmul(x1, x2, dtype, *, bias=None, x1_scale=None, x2_scale=None, group_sizes=None, perm_x1=None, perm_x2=None, perm_y=None, batch_split_factor=1, x1_dtype=None, x2_dtype=None) -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. First matrix in the matrix multiplication. The data layout can be `ND`. Only 3D input is supported, with shape `(m, b, k)`. The data type can be `torch.float8_e5m2`, `torch.float8_e4m3fn`, or `torch_npu.hifloat8`.
- **`x2`** (`Tensor`): Required. Second matrix in the matrix multiplication. The data layout can be `ND` or `NZ`. Only 3D input is supported, with shape `(b, k, n)` or `(b, n, k)`. The size of the `k` dimension of `x2` must be the same as that of `x1`. The data type can be `torch.float8_e5m2`, `torch.float8_e4m3fn`, or `torch_npu.hifloat8`.
- **`dtype`** (`int`): Required. Data type of the output. Supported values are `torch.float16`, `torch.bfloat16`, and `torch_npu.hifloat8`.
- \*: Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`bias`** (`Tensor`): Optional. Bias matrix for the matrix multiplication. This parameter is not supported in the current version. Use the default value.
- **`x1_scale`** (`Tensor`): Optional. Quantization scale for the left matrix. The data layout can be `ND`. The data type can be `torch.float32`, `torch.float8_e8m0fnu`, or `torch.int64`. The shape can be 1D or 4D.
- **`x2_scale`** (`Tensor`): Optional. Quantization scale for the right matrix. The data layout can be `ND`. The data type can be `torch.float32`, `torch.float8_e8m0fnu`, or `torch.int64`. The shape can be 1D or 4D.
- **`group_sizes`** (`List[int]`): Optional. Quantization group sizes. The data type is `torch.int32`. The default value is `None`.
  - Only a 3D list is supported, in the form of [group\_m, group\_n, group\_k], which respectively indicate the quantization grouping on the m, n, and k dimensions. Taking group\_m as an example, it indicates that every group\_m values on the m dimension correspond to one quantization parameter.
  - When one or more of [group\_m, group\_n, group\_k] are 0, the API resets the value according to the input shapes of `x1`, `x2`, `x1_scale`, and `x2_scale`. Computation principle: If group\_m=0, the quantization group value in the m direction is inferred by the API. The inference formula is group\_m=m/scale\_m (m must be divisible by scale\_m), where m is the same as m in the `x1` shape and scale_m is the same as m in the `x1_scale` shape.
  - This parameter is required only in the mx quantization mode. Currently, [group\_m, group\_n, group\_k] supports only [0,0,32], [0,1,32], [1,0,32], and [1,1,32].

- **`perm_x1`** (`List[int]`): Optional. Transpose sequence for the first matrix in the matrix multiplication. The size is `3`, and the data type is `torch.int64`. The data layout can be `ND`. Only `[1, 0, 2]` is supported.
- **`perm_x2`** (`List[int]`): Optional. Transpose sequence for the second matrix in the matrix multiplication. The size is `3`, and the data type is `torch.int64`. The data layout can be `ND`. `[0, 1, 2]` and `[0, 2, 1]` are supported.
- **`perm_y`** (`List[int]`): Optional. Transpose sequence for the output matrix of the matrix multiplication. The size is `3`, and the data type is `torch.int64`. The data layout can be `ND`. Only `[1, 0, 2]` is supported.
- **`batch_split_factor`** (`int`): Optional. Specifies the split size of the `b` dimension of the output matrix of the matrix multiplication. The data type is `int32`. The default value is `1`. Currently, only `1` is supported.
- **`x1_dtype`** (`int`): Optional. Data type of `x1`. Supported values are `torch.float8_e5m2`, `torch.float8_e4m3fn`, and `torch_npu.hifloat8`.
- **`x2_dtype`** (`int`): Optional. Data type of `x2`. Supported values are `torch.float8_e5m2`, `torch.float8_e4m3fn`, and `torch_npu.hifloat8`.

## Return Values

**`y`** (`Tensor`): final computation result, $out$ in the formulas. The data layout can be ND, and the shape can be 3D. The shape is (m, b, n). The data type can be `torch.float16`, `torch.bfloat16`, or `torch_npu.hifloat8`.

## Constraints

- This API can be used in training and inference scenarios.
- This API supports single-operator mode and graph mode.
- Graph mode supports the K-C and mx quantization modes.
- In the K-C quantization scenario:
  - `x1_scale` and `x2_scale` support only 1D input. The shape of `x1_scale` must be (m,), and the shape of `x2_scale` must be (n,).
  - `x2` supports only the ND format input.
  - Only 512 is supported for k, and only 128 is supported for n.
  - `perm_x2` supports only [0, 1, 2].

- In the mx quantization scenario:
  - `x1` and `x2` support only `torch.float8_e4m3fn` input. Only multiples of 64 are supported for k.
  - `x1_scale` and `x2_scale` support only 4D input. The shape of `x1_scale` must be (m, b, k/64, 2). When `perm_x2` is [0,1,2], the shape of `x2_scale` must be (b, k/64, n, 2); when `perm_x2` is [0,2,1], the shape of `x2_scale` must be (b, n, k/64, 2).
  - Converting `x2` to an Ascend-friendly data layout is supported to improve the transfer efficiency. Call `torch_npu.npu_format_cast` to convert the input `x2` (weight) to an Ascend-friendly data layout.

- In the T-C quantization scenario:
  - `x1` and `x2` support only `torch_npu.hifloat8` input.
  - `x2` supports only the ND format input.
  - `x1_scale` and `x2_scale` support only 1D input. `x1_scale` can be empty; when it is not empty, its shape must be (1,). The shape of `x2_scale` must be (n,).

## Example

- Single-operator call
  - K-C quantization scenario:

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

- Graph mode call
  - K-C quantization scenario:

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    M, K, N, Batch = 32, 512, 128, 32
    x1 = torch.randint(-5, 5, (M, Batch, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    x2 = torch.randint(-5, 5, (Batch, K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()

    x1_scale = torch.randint(-3, 3, (M, ), dtype=torch.float32).npu()
    x2_scale = torch.randint(-3, 3, (N, ), dtype=torch.float32).npu()

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2, x1_scale, x2_scale, dtype, perm_x1, perm_x2, perm_y, batch_split_factor=1):
            output = torch_npu.npu_transpose_quant_batchmatmul(x1, x2, dtype, x1_scale=x1_scale, x2_scale=x2_scale, perm_x1=perm_x1, perm_x2=perm_x2, perm_y=perm_y, batch_split_factor=batch_split_factor)
            return output

    model = MyModel().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    output = model(x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), torch.float16, (1, 0, 2), (0, 1, 2), (1, 0, 2)).to("cpu")
    ```
