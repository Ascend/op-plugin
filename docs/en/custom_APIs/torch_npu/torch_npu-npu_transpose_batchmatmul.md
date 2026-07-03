# torch\_npu.npu\_transpose\_batchmatmul<a name="en-us_topic_0000002350565344"></a>

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |

## Function<a name="en-us_topic_0000002319693140_section14441124184110"></a>

- Description: Performs batch matrix multiplication between the `input` and `weight` tensors. Only three-dimensional tensors are supported. Tensors can be transposed based on the input index arrays. `perm_x1` and `perm_x2` represent the permutation sequences for tensors `input` and `weight`, respectively. The dimension with a sequence value of `0` represents the batch dimension, while the remaining two dimensions are used for matrix multiplication.
- Formulas:
    - Non-quantized fusion:
    $$
    Y = (input^{T_1} @ weight^{T_2} + bias)^{T_y}
    $$

    - Quantized fusion:
    $$
    Y = (input^{T_1} @ weight^{T_2} + bias)^{T_y} * scale
    $$    

    $T_1$, $T_2$, and $T_y$ represent the permutation sequences specified by the parameters `perm_x1`, `perm_x2`, and `perm_y`, respectively.

## Prototype<a name="en-us_topic_0000002319693140_section45077510411"></a>

```python
torch_npu.npu_transpose_batchmatmul(input, weight, *, bias=None, scale=None, perm_x1=[0,1,2], perm_x2=[0,1,2], perm_y=[1,0,2], batch_split_factor=1) -> Tensor
```

## Parameters<a name="en-us_topic_0000002319693140_section112637109429"></a>

- **`input`** (`Tensor`): Required. The first input matrix for matrix multiplication. The data type can be `float16`, `bfloat16`, or `float32`. The size of the -1 dimension (the last dimension) must be $\le 65535$. The data layout can be ND. This parameter must be 3D with shape `(B, M, K)` or `(M, B, K)`, where the value range of `B` is [1, 65536). Non-contiguous tensors are supported.
- **`weight`** (`Tensor`): Required. The second input matrix for matrix multiplication. The data type can be `float16`, `bfloat16`, or `float32`. The size of the -1 dimension (the last dimension) must be $\le 65535$. The data layout can be ND. This parameter must be 3D with shape `(B, K, N)`, where the value range of `N` is `[1, 65536)`. Non-contiguous tensors are supported. The reduction dimension of `weight` must match the size of the reduction dimension of `input`.
- **`bias`** (`Tensor`): Optional. The bias matrix for matrix multiplication. This parameter is currently not supported. Use its default value.
- **`scale`** (`Tensor`): Optional. Quantization input matrix. The data type can be `int64` and `uint64`. The data layout can be ND. This parameter must be 1D with shape `(B*N,)`, where the value range of `B*N` is [1, 65536). Non-contiguous tensors are supported.
- **`perm_x1`** (`List[int]`): Optional. The permutation sequence for transposing the first matrix. The size must be 3. The data type is `int64`. The data layout can be ND. Valid values are `[0, 1, 2]` or `[1, 0, 2]`.
- **`perm_x2`** (`List[int]`): Optional. The permutation sequence for transposing the second matrix. The size must be 3. The data type is `int64`. The data layout can be ND. Only `[0, 1, 2]` is supported.
- **`perm_y`** (`List[int]`): Optional. The permutation sequence for transposing the output matrix. The size must be 3. The data type is `int64`. The data layout can be ND. Only `[1, 0, 2]` is supported.
- **`batch_split_factor`** (`int`): Optional. The split size of the $N$ dimension in the output matrix. The data type can be `int32`. The value range is [1, N] and the value must evenly divide $N$. The default value is `1`. Note: When `scale` is specified, `batch_split_factor` must be `1`.

## Return Values<a name="en-us_topic_0000002319693140_section22231435517"></a>

`Tensor`

The final computation result, $Y$ in the formulas. The data layout can be ND. This parameter must be 3D.

- When the input `scale` is provided, the data type must be `int8`, and the shape is `(M, 1, B * N)`. Otherwise, the data type can be `float16`, `bfloat16`, or `float32`.
- When `batch_split_factor` > 1, the shape is `[batch_split_factor, M, B * N/batch_split_factor]`.

## Constraints<a name="en-us_topic_0000002319693140_section12345537164214"></a>

- This API can be used in inference scenarios.
- This API supports both single-operator mode and graph mode.
- Variables used in input parameter tensor shapes:
    - When `perm_x1` is `[1, 0, 2]` (the `input` matrix requires transposition), the value range of $K \times B$ is [1, 65536). When `perm_x1` is `[0, 1, 2]`, $K$ must be less than `65536`.
    - Both $K$ and $N$ must be evenly divisible by `16`.

## Examples<a name="en-us_topic_0000002319693140_section14459801435"></a>

- Single-operator call

    ```python
    import torch
    import torch_npu
    M, K, N, Batch = 32, 512, 128, 16
    x1 = torch.randn((M, Batch, K), dtype=torch.float16)
    x2 = torch.randn((Batch, K, N), dtype=torch.float16)
    batch_split_factor=1
    output = torch_npu.npu_transpose_batchmatmul(x1.npu(), x2.npu(), bias=None, scale=None, perm_x1=(1,0,2), perm_x2=(0,1,2), perm_y=(1,0,2), batch_split_factor=batch_split_factor)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    M, K, N, Batch = 32, 512, 128, 16
    x1 = torch.randn((M, Batch, K), dtype=torch.float16)
    x2 = torch.randn((Batch, K, N), dtype=torch.float16)

    class MyModel1(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2, perm_x1, perm_y, batch_split_factor=1):
            output = torch_npu.npu_transpose_batchmatmul(x1, x2, perm_x1=perm_x1, perm_y=perm_y, batch_split_factor=batch_split_factor)
            output = output.add(1)
            return output

    model = MyModel1().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    output = model(x1.npu(), x2.npu(), (1, 0, 2), (1, 0, 2)).to("cpu")
    ```
