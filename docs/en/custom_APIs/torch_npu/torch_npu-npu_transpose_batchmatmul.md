# torch_npu.npu_transpose_batchmatmul

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id3 -->

## Function

- **Function**: performs the matrix multiplication of the tensor `input` and the tensor `weight`. Only 3D Tensors are supported. The Tensors support transposition, and the transposition sequence changes according to the passed sequence. `perm_x1` indicates the transposition sequence of the tensor `input`, and `perm_x2` indicates the transposition sequence of the tensor `weight`. The dimension with the sequence value 0 is the batch dimension, and the matrix multiplication is performed on the other two dimensions.

- **Formulas**:

  - Non-quantization fusion:

    $$
    Y = (input^{T_1} @ weight^{T_2} + bias)^{T_y}
    $$

  - Quantization fusion:

    $$
    Y = (input^{T_1} @ weight^{T_2} + bias)^{T_y} * scale
    $$

    T1, T2, and Ty describe the transposition sequences through the parameters `perm_x1`, `perm_x2`, and `perm_y` respectively.

## Prototype

```python
torch_npu.npu_transpose_batchmatmul(input, weight, *, bias=None, scale=None, perm_x1=[0,1,2], perm_x2=[0,1,2], perm_y=[1,0,2], batch_split_factor=1) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. The first input matrix for matrix multiplication. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The size of the -1 dimension (the last dimension) must be $\le 65535$. The data layout can be ND. This parameter must be 3D with shape `(B, M, K)` or `(M, B, K)`, where the value range of `B` is [1, 65536). Non-contiguous tensors are supported.

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the last axis (-1 axis) <= 65535, and the value range of B is [1, 65536).
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT products</term>: the shape can be 3D (B, M, K) or (M, B, K).
  <!-- end id5 -->

- **`weight`** (`Tensor`): Required. The second input matrix for matrix multiplication. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The size of the -1 dimension (the last dimension) must be $\le 65535$. The data layout can be ND. This parameter must be 3D with shape `(B, K, N)`, where the value range of `N` is `[1, 65536)`. Non-contiguous tensors are supported. The reduction dimension of `weight` must match the size of the reduction dimension of `input`.

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the last axis (-1 axis) <= 65535. The shape can be 3D (B, K, N), and the value range of N is [1, 65536).
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: the shape can be 3D (B, K, N) or (B, N, K).
  <!-- end id7 -->

- \*: Variables before it support positional input or key-value assignment; variables after it support only key-value assignment. A variable with a default value uses the default value if not assigned, and a variable without a default value must be assigned.
- **`bias`** (`Tensor`): Optional. The bias matrix for matrix multiplication. This parameter is currently not supported. Use its default value.
- **`scale`** (`Tensor`): Optional. Quantization scale factor. The data type can be `torch.int64` and `torch.uint64`. The data layout can be ND. This parameter must be 1D with shape `(B*N,)`, where the value range of `B*N` is [1, 65536). Non-contiguous tensors are supported.

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the value range of B*N is [1, 65536).
  <!-- end id8 -->

- **`perm_x1`** (`List[int]`): Optional. The permutation sequence for transposing the first matrix. The size must be 3. The data type is `torch.int64`. The data layout can be ND. Valid values are `[0, 1, 2]` or `[1, 0, 2]`.
- **`perm_x2`** (`List[int]`): Optional. The permutation sequence for transposing the second matrix. The size must be 3. The data type is `torch.int64`. The data layout can be ND. Only `[0, 1, 2]` is supported.

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: only [0, 1, 2] is supported.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: [0, 1, 2] and [0, 2, 1] are supported.
  <!-- end id10 -->

- **`perm_y`** (`List[int]`): Optional. The permutation sequence for transposing the output matrix. The size must be 3. The data type is `torch.int64`. The data layout can be ND. Only `[1, 0, 2]` is supported.
- **`batch_split_factor`** (`int`): Optional. Split size of the $N$ dimension in the output matrix of the matrix multiplication. The data type can be `int32`. The value range is [1, N], and $N$ must be divisible by this value. The default value is `1`. Note: When `scale` is provided, `batch_split_factor` must be 1.

## Return Values

- **`y`**(`Tensor`): final computation result. The data layout can be $ND$, and the shape can be 3D.

  - When the input `scale` has a value, the data type can only be `torch.int8`, and the shape is (M, 1, B*N); otherwise, the data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
  - When `batch_split_factor`>1, the shape size is calculated as [batch_split_factor, M, B*N/batch_split_factor].

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator mode and TorchAir graph mode.

<!-- npu="A3,910b" id11 -->
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
  - When `perm_x1` is [1, 0, 2], that is, when the `input` matrix must be transposed, the value range of K*B is [1, 65536); when `perm_x1` is [0, 1, 2], K must be less than 65536.
  - K and N must be divisible by 16.
<!-- end id11 -->

## Example

- Single-operator call

    ```python
    import torch
    import torch_npu
    M, K, N, Batch = 32, 512, 128, 16
    x1 = torch.randn((M, Batch, K), dtype=torch.float16)
    x2 = torch.randn((Batch, K, N), dtype=torch.float16)
    batch_split_factor = 1
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
