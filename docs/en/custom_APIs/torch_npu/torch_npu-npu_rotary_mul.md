# torch_npu.npu_rotary_mul

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

- Description: Implements Rotary Position Embedding (RoPE) by injecting positional information into input features through two-dimensional plane rotations.
- Formulas:
     $$
    output = x * cos + rotate(x) * sin
     $$
     $x$ represents the input `input`, while $cos$ and $sin$ represent the rotation coefficient inputs `r1` and `r2`, respectively. The `rotate(x)` operation supports the following computation modes.

     - When `rotary_mode` is `"half"`, the input vector is split into two halves along the last dimension, and the rotation is applied:
         $$
         x_1, x_2 = chunk(x,2,dim=-1)\\
         rotate(x) = concat(-x_2,x_1)
         $$

     - When `rotary_mode` is `"interleave"`, the input vector is processed in an interleaved manner, and the rotation is applied:
         $$
         x_1 = x[..., ::2], x_2 = x[..., 1::2]\\
         rotate(x) = rearrange(torch.stack((-x_2, x_1), dim=-1), "... d two -> ...(d two)", two=2)\\
         $$
     - When `rotate` is provided, the rotation matrix is generated using the following formula:
         $$
         rotate(x) = x \cdot rotate\\
         $$

- Equivalent computation logic:
    
     The `fused_rotary_position_embedding` operator can be used as an equivalent replacement for `torch_npu.npu_rotary_mul`. The computation logic of the two operators is identical.
     
     ```python
     import torch
     from einops import rearrange
     
     # mode = 0
     def rotate_half(x):
         x1, x2 = torch.chunk(x, 2, dim=-1)
         return torch.cat((-x2, x1), dim=-1)
     
     # mode = 1
     def rotate_interleaved(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ...(d two)", two=2)
     
     def fused_rotary_position_embedding(x, cos, sin, interleaved=False):
         if not interleaved:
             return x * cos + rotate_half(x) * sin
         else:
             return x * cos + rotate_interleaved(x) * sin
     ```     

## Prototype

```python
torch_npu.npu_rotary_mul(input, r1, r2, rotary_mode='half', rotate=None) -> Tensor
```

> [!NOTE]  
> In model training scenarios, the forward input `input` is retained for backward computation. In scenarios where backward gradients are not required for `r1` and `r2` (`requires_grad=False`), using this API causes device memory usage to increase compared with the unfused operators. Backward propagation is not supported when the `rotate` parameter is provided.

## Parameters

- **`input`** (`Tensor`): Required. Input tensor. The shape must have 3 or 4 dimensions. The data type can be `float16`, `bfloat16`, or `float32`.
- **`r1`** (`Tensor`): Required. Cosine rotation coefficient. The shape must have 3 or 4 dimensions. The data type can be `float16`, `bfloat16`, or `float32`.
- **`r2`** (`Tensor`): Required. Sine rotation coefficient. The shape must have 3 or 4 dimensions. The data type can be `float16`, `bfloat16`, or `float32`.
- **`rotary_mode`** (`str`): Optional. Computation mode. Valid values are `half` and `interleave`. The default value is `half`.
- **`rotate`** (`Tensor`): Optional. Equivalent transformation matrix used to apply positional transformations to `input`. The shape must have 2 dimensions. The data type can be `float16`, `bfloat16`, or `float32`. For construction details, see the examples.

## Return Values

`Tensor`

Output tensor representing the computation result. Its shape and data type must be identical to those of `input`.

## Constraints

- `jit_compile=False` scenarios (applicable to <term>Atlas A2 training products</term> or <term>Atlas A3 training products</term>):
    - When `rotary_mode` is `"half"`:

        `input`: The layout can be `"BNSD"`, `"BSND"`, `"SBND"`, or `"TND"`. The `D` dimension size must be less than `896` and must be divisible by `2`. The sizes of the `B` and `N` dimensions must be less than `1000`. When backward gradients are required for `cos` or `sin`, the product of `B` and `N` must be less than or equal to `1024`.

        `r1` or `r2`: The data range is [-1, 1]. The supported `input` layouts are as follows:

        - When `x` is `"BNSD"`: `"11SD"`, `"B1SD"`, or `"BNSD"`
        - When `x` is `"BSND"`: `"1S1D"`, `"BS1D"`, or `"BSND"`
        - When `x` is `"SBND"`: `"S11D"`, `"SB1D"`, or `"SBND"`
        - When `x` is `"TND"`: `"T1D"` or `"TND"`

            > [!NOTICE]  
            > When `rotary_mode` is `"half"`, when the input layout configuration is `BNSD` and `D` is not 32-byte aligned, do not use this fused operator. That is, do not enable the `--use-fused-rotary-pos-emb` option in the model startup script. Otherwise, performance degradation can occur.

    - When `rotary_mode` is `"interleave"`:

        **`input`**: The layout can be `"BNSD"`, `"BSND"`, `"SBND"`, or `"TND"`. The product of `B` and `N` must be less than `1000`. The `D` dimension size must be less than `896` and must be divisible by `2`.

        `r1` or `r2`: The data range is [-1, 1]. The supported `input` layouts are as follows:

        - When `x` is `"BNSD"`: `"11SD"`
        - When `x` is `"BSND"`: `"1S1D"`
        - When `x` is `"SBND"`: `"S11D"`
        - When `x` is `"TND"`: `"T1D"`

- **`jit_compile=True`** scenarios (applicable to <term>Atlas training products</term>, <term>Atlas A2 training products</term>, and <term>Atlas inference products</term>):

     Only `rotary_mode="half"` is supported. Typical `r1` and `r2` layouts are `"11SD"`, `"1S1D"`, and `"S11D"`.

     The input must be a 4D tensor. The sizes of the `B` and `N` dimensions must be less than or equal to `1000`. The size of the $D$ dimension must be `128`.

     In broadcasting scenarios, the cumulative data volume across the broadcasting axes must not exceed `1024`.

- Recommended usage scenarios for `rotate`:
  - When `rotary_mode` is `"interleave"`:
  - When `rotary_mode` is `"half"`, this configuration is recommended only when the input tensor needs to be split into multiple segments along the last dimension. In this configuration, a rotary encoding matrix can be constructed so that all rotary position encoding computations are completed in a single call to improve performance. For example, assume `x` uses the `"BSND"` layout and is split into three segments:
     $x = [x1|x2|x3]_{(dim=4)} ∈ R^{B×S×N×D}, x1 ∈ R^{B×S×N×D1}, x2 ∈ R^{B×S×N×D2}, x3 ∈ R^{B×S×N×D3}$, where $D = D1 + D2 + D3$.
     The `rotate` matrix can then be constructed as follows to perform rotary position encoding on `x` in a single call:
     $$rotate = diag(rotate1, rotate2, rotate3) = \begin{pmatrix}rotate1&0&0\\0&rotate2&0\\0&0&rotate3\\\end{pmatrix}$$
     `rotate1`, `rotate2`, and `rotate3` represent the rotary encoding matrices corresponding to `x1`, `x2`, and `x3`, respectively. For details about constructing a single rotary encoding matrix, see the examples.

## Examples

- 4D input example:

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.rand(2, 2, 5, 128).npu()
    >>> r1 = torch.rand(1, 2, 1, 128).npu()
    >>> r2 = torch.rand(1, 2, 1, 128).npu()
    >>> out = torch_npu.npu_rotary_mul(x, r1, r2)
    >>> print(out.shape)
    torch.Size([2, 2, 5, 128])
    >>> print(out)
    tensor([[[[ 0.1017, -0.0871,  0.2722,  ...,  0.4668,  0.4320,  0.4252],
            [ 0.2908, -0.0068,  0.4026,  ...,  0.1540,  0.2653,  0.6754],
            [ 0.1124, -0.0637,  0.0834,  ...,  0.5127,  0.1423,  0.0636],
            [ 0.1014,  0.0129,  0.3392,  ...,  0.7390,  0.7147,  0.1751],
            [ 0.3266, -0.0177,  0.2263,  ...,  0.9936,  0.3717,  0.3403]],

            [[ 0.1999, -0.5646,  0.0910,  ...,  0.1747,  0.3801,  0.0675],
            [ 0.2688,  0.3714,  0.2647,  ...,  0.0769,  0.0481,  0.1988],
            [ 0.1404,  0.1749,  0.4082,  ...,  0.2291,  0.5246,  0.0615],
            [-0.4368,  0.2962,  0.2655,  ...,  0.0284,  0.5518,  0.2853],
            [ 0.0812,  0.4214,  0.4906,  ...,  0.1684,  0.5756,  0.2966]]],


            [[[ 0.3887, -0.0777,  0.0328,  ...,  0.4946,  0.5197,  0.8397],
            [ 0.0283, -0.0858,  0.2244,  ...,  0.2542,  0.3899,  0.8239],
            [ 0.1993, -0.0765,  0.2022,  ...,  0.7701,  0.6514,  0.0557],
            [ 0.1424, -0.0795,  0.4005,  ...,  0.3839,  0.5843,  0.2539],
            [ 0.2812, -0.0479,  0.1748,  ...,  0.6403,  0.5840,  0.3274]],

            [[ 0.1308, -0.2528,  0.6242,  ...,  0.2614,  0.4986,  0.0893],
            [ 0.3121,  0.1706,  0.6207,  ...,  0.0731,  0.1644,  0.2398],
            [ 0.3232,  0.0695,  0.2875,  ...,  0.1104,  0.3334,  0.2233],
            [ 0.4909,  0.3554,  0.8431,  ...,  0.2265,  0.4873,  0.3106],
            [-0.2269, -0.1447, -0.0395,  ...,  0.1374,  0.2142,  0.3628]]]],
        device='npu:0')
    ```

- 3D input example:

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.rand(2, 5, 128).npu()
    >>> r1 = torch.rand(2, 1, 128).npu()
    >>> r2 = torch.rand(2, 1, 128).npu()
    >>> out = torch_npu.npu_rotary_mul(x, r1, r2, "half")
    >>> print(out)
    tensor([[[-0.1200, -0.2515, -0.3189,  ...,  0.2283,  1.1038,  0.3439],
            [ 0.1083,  0.0257,  0.1864,  ...,  0.5940,  0.8644,  0.5961],
            [-0.0147, -0.1542,  0.0516,  ...,  0.7441,  0.2782,  0.4797],
            [-0.0601, -0.0338, -0.3731,  ...,  0.9809,  0.7416,  0.4876],
            [ 0.1785, -0.0542, -0.3634,  ...,  0.5057,  0.7511,  1.3088]],

                [[ 0.0076,  0.0931, -0.4161,  ...,  0.4964,  0.2680,  0.1291],
                [-0.2149,  0.1523, -0.0274,  ...,  0.1997,  0.8318,  0.2630],
                [ 0.1087,  0.4846,  0.0684,  ...,  0.0183,  0.9503,  0.0555],
                [-0.1946,  0.6020, -0.6751,  ...,  0.8629,  0.5454,  0.1392],
                [ 0.0772,  0.5112, -0.4875,  ...,  0.7065,  0.6798,  0.1513]]],
            device='npu:0')
    ```

- Example of `rotate` matrix generation:

    ```python
    import torch
    import torch_npu

    def get_interleave_matrix(n):
        matrix = torch.zeros(n, n, dtype=torch.bfloat16)
        for i in range(0, n, 2):
            matrix[i + 0, i + 1] = 1
            matrix[i + 1, i + 0] = 1
        return matrix

    def get_half_matrix(n):
        matrix = torch.zeros(n, n, dtype=torch.bfloat16)
        half = n // 2
        matrix[:half, half:] = torch.eye(half)
        matrix[half:, :half] = -torch.eye(half)
        return matrix

    def compose_2matrix(A, B):
        total_rows = A.size(0) + B.size(0)
        total_cols = A.size(1) + B.size(1)
        result = torch.zeros(total_rows, total_cols, dtype=torch.bfloat16)
        result[:A.size(0), :A.size(1)] = A
        b_row_start = A.size(0)
        b_col_start = A.size(1)
        result[b_row_start:b_row_start + B.size(0), 
            b_col_start:b_col_start + B.size(1)] = B
        return result

    def main():
        # interleave
        inter_mat_128 = get_interleave_matrix(128)
        inter_mat_64 = get_interleave_matrix(64)
        # interleave 2D
        inter_mat_128_64 = compose_2matrix(inter_mat_128, inter_mat_64)

        x = torch.rand(2, 2, 5, 128).npu()
        r1 = torch.rand(1, 2, 1, 128).npu()
        r2 = torch.rand(1, 2, 1, 128).npu()
        out = torch_npu.npu_rotary_mul(x, r1, r2, "interleave", inter_mat_128.npu())
    ```
