# torch_npu.npu_rotary_mul

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->

## Function

- Function: implements Rotary Position Embedding (RoPE), which injects position information by performing a 2D planar rotation on the input features. FakeTensor mode is supported.
- Formulas:
     $$
    output = x * cos + rotate(x) * sin
     $$
     In the formula, $x$ is the input `input`, $cos$ and $sin$ are the rotation coefficient inputs `r1` and `r2` respectively, and `rotary_mode` supports the following computation modes:

     - When `rotary_mode` is `half`, the input vector is divided into two halves along the last dimension, and then the rotation is applied:
         $$
         x_1, x_2 = chunk(x,2,dim=-1)\\
         rotate(x) = concat(-x_2,x_1)
         $$

     - When `rotary_mode` is `interleave`, the input vector is processed in an interleaved order, and then the rotation is applied:
         $$
         x_1 = x[..., ::2], x_2 = x[..., 1::2]\\
         rotate(x) = rearrange(torch.stack((-x_2, x_1), dim=-1), "... d two -> ...(d two)", two=2)\\
         $$
     - When the `rotate` parameter is passed, the rotation matrix is generated as follows:
         $$
         rotate(x) = x \cdot rotate\\
         $$

     <!-- npu="950" id6 -->
     - <term>Ascend 950PR&950DT products</term>: when `rotary_mode` is `quarter`, the input is divided into four parts $x_1,x_2,x_3,x_4$ along the last dimension:
         $$
         \begin{aligned}
         x_1 &= x[..., :D/4], & x_2 &= x[..., D/4:D/2], \\
         x_3 &= x[..., D/2:3D/4], & x_4 &= x[..., 3D/4:], \\
         x_{rotate} &= concat(-x_2,x_1,-x_4,x_3), \\
         y &= x \cdot r1 + x_{rotate} \cdot r2
         \end{aligned}
         $$
     <!-- end id6 -->
     <!-- npu="950" id7 -->
     - <term>Ascend 950PR&950DT products</term>: when `rotary_mode` is `interleave-half`, the elements at even and odd positions are first concatenated into `x_part1` and `x_part2` respectively:
         $$
         \begin{aligned}
         x_1 &= x[..., ::2], & x_2 &= x[..., 1::2], \\
         x_{part1} &= concat(x_1,x_2), \\
         x_{part2} &= concat(-x_2,x_1), \\
         y &= x_{part1} \cdot r1 + x_{part2} \cdot r2
         \end{aligned}
         $$
     <!-- end id7 -->

- Equivalent computation logic:
    
     `fused_rotary_position_embedding` can be used as an equivalent replacement of `torch_npu.npu_rotary_mul`, and the computation logic of the two is the same.
     
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
> In model training scenarios, the input `input` of the forward operator is retained for use in the backward computation. When the backward gradient of `r1` and `r2` does not need to be computed (`requires_grad=False`), using this API increases the device memory usage compared with the small operators before fusion. Backward propagation is not supported when the `rotate` parameter is passed.

## Parameters

- **`input`** (`Tensor`): Required. Input tensor. The shape must have 3 or 4 dimensions. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
- **`r1`** (`Tensor`): Required. Cosine rotation coefficient. The shape must have 3 or 4 dimensions. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
- **`r2`** (`Tensor`): Required. Sine rotation coefficient. The shape must have 3 or 4 dimensions. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
- **`rotary_mode`** (`str`): Optional. Computation mode. Valid values are `half` and `interleave`. The default value is `half`.
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term> also support the `quarter` and `interleave-half` modes.
  <!-- end id8 -->
- **`rotate`** (`Tensor`): Optional. Equivalent transformation matrix used to apply positional transformations to `input`. The shape must have 2 dimensions. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. For construction details, see the examples.

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>: 2D inputs are supported, and the data types can be `torch.float16`, `torch.bfloat16`, and `torch.float32`.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>: this parameter is not supported. Use the default value `None` or omit this parameter.
    <!-- end id10 -->

## Return Values

`Tensor`

Output computation result. The shape and dtype must be the same as those of `input`.

## Constraints

<!-- npu="A3,910b" id11 -->
- **`jit_compile=False` scenario** (applicable to <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>):
    - half mode:

        `input`: supported layouts: $BNSD, BSND, SBND, TND; D < 896$ and D is a multiple of 2; $B, N < 1000$; when the backward gradient of $cos/sin$ must be computed, $B*N <= 1024$.

        `r1, r2`: data range: [-1, 1]. Support of the corresponding `input` layouts:

        - x is $BNSD$: $11SD, B1SD, BNSD$;
        - x is $BSND$: $1S1D, BS1D, BSND$;
        - x is $SBND$: $S11D, SB1D, SBND$;
        - x is $TND$: $T1D, TND$.

            > [!NOTICE]  
            > In half mode, when the input layout is $BNSD$ and $D$ is not 32-byte aligned, using this fusion operator is not recommended (do not enable the `--use-fused-rotary-pos-emb` option in the model startup script); otherwise, the performance may decrease.

    - interleave mode:

        `input`: supported layouts: $BNSD, BSND, SBND, TND; B*N < 1000; D < 896$ and $D$ is a multiple of 2;

        `r1, r2`: data range: [-1, 1]. Support of the corresponding `input` layouts:

        - x is $BNSD: 11SD$;
        - x is $BSND: 1S1D$;
        - x is $SBND: S11D$;
        - x is $TND: T1D$.
<!-- end id11 -->
<!-- npu="910b,910,310p" id12 -->
- **`jit_compile=True` scenario** (applicable to <term>Atlas training products</term>, <term>Atlas A2 training products</term>, and <term>Atlas inference products</term>):

     Only the half mode is supported for `rotary_mode`, and the `r1/r2` layouts are generally $11SD, 1S1D, S11D$.

     The input shape must be 4D, where the values of the $B$ and $N$ dimensions must be less than or equal to 1000 and the value of the $D$ dimension is 128.

     In broadcast scenarios, the total data volume of the broadcast axes must not exceed 1024.
<!-- end id12 -->
<!-- npu="950" id13 -->
- <term>Ascend 950PR&950DT products</term>: \(B, S, N, D\) is used to represent the shape of the 4D input `input`. The detailed constraints of each parameter are as follows.
    - The size of the D dimension of the input tensors `input`, `r1`, and `r2` and the output tensor must be the same and must be less than or equal to 1024. In half, interleave, and interleave-half modes, D must be divisible by 2; in quarter mode, D must be divisible by 4.
    - The shapes of the input tensors `r1` and `r2` must be exactly the same and must satisfy one of the following conditions:
        - the shape is (1, 1, 1, D);
        - the shape is (B, S, N, D);
        - the shape is (B, 1, N, D) or (B, S, 1, D);
        - the shape is (1, 1, N, D), (1, S, 1, D), or (B, 1, 1, D).
    - When `input` is an empty Tensor, the output is also an empty Tensor and is not subject to the preceding shape constraints.
<!-- end id13 -->
<!-- npu="A3,910b" id14 -->
- Recommended scenarios for rotate (applicable to <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>):
  - interleave mode
  - The half mode is recommended only in the following scenario: when the input matrix x must be split into multiple parts along the last dimension, a rotation encoding matrix can be constructed to obtain a performance benefit with a single call. Taking the layout of x as BSND and splitting into 3 parts as an example:
     x is split into 3 parts, $x = [x1|x2|x3]_{(dim=4)} \in R^{B\times S\times N\times D}, x1 \in R^{B\times S\times N\times D1},x2 \in R^{B\times S\times N\times D2},x3 \in R^{B\times S\times N\times D3}$, where D = D1 + D2 + D3,
     Then a rotate matrix can be constructed to complete the rotary position embedding computation of x in a single call. The rotate matrix is constructed as follows:
     $$rotate = diag(rotate1, rotate2, rotate3) = \begin{pmatrix}rotate1&0&0\\0&rotate2&0\\0&0&rotate3\\\end{pmatrix}$$
     rotate1, rotate2, and rotate3 are the rotation encoding matrices of x1, x2, and x3 respectively. For the construction of a single rotation matrix, see the example.
<!-- end id14 -->

## Example

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

<!-- npu="950" id15 -->
- <term>Ascend 950PR&950DT products</term> call example:

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.rand(2, 5, 128).npu()
    >>> r1 = torch.rand(2, 1, 128).npu()
    >>> r2 = torch.rand(2, 1, 128).npu()
    >>> out = torch_npu.npu_rotary_mul(x, r1, r2, "half")
    ```
<!-- end id15 -->
<!-- npu="A3,910b" id16 -->
- rotate generation example (applicable only to <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>):

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
<!-- end id16 -->
<!-- npu="950" id17 -->
- Graph mode call: Only applicable to <term>Ascend 950PR&950DT products</term>.

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, r1, r2):
            return torch_npu.npu_rotary_mul(x, r1, r2, rotary_mode="half")

    model = Model().npu()
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend)

    x = torch.rand((2, 1, 1, 64), dtype=torch.float16).npu()
    r1 = torch.rand((2, 1, 1, 64), dtype=torch.float16).npu()
    r2 = torch.rand((2, 1, 1, 64), dtype=torch.float16).npu()
    result = model(x, r1, r2)
    ```
<!-- end id17 -->
