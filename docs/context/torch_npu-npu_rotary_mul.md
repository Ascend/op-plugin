# torch_npu.npu_rotary_mul

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term> | √   |

## 功能说明

-   API功能：实现Rotary Position Embedding (RoPE) 旋转位置编码，通过对输入特征进行二维平面旋转注入位置信息。
-   计算公式：
     $$
    output = x * cos + rotate(x) * sin
     $$
     其中$x$是输入`input`，$cos$和$sin$分别是旋转系数输入`r1`和`r2`，输入rotate支持两种计算模式：

     - 当rotary_mode='half'时，将输入向量沿最后一个维度分为两半，然后应用旋转：
         $$
         x_1, x_2 = chunk(x,2,dim=-1)\\
         rotate(x) = concat(-x_2,x_1)
         $$

     - 当rotary_mode='interleave'时，将输入向量按交错顺序处理，然后应用旋转：
         $$
         x_1 = x[..., ::2], x_2 = x[..., 1::2]\\
         rotate(x) = rearrange(torch.stack((-x_2, x_1), dim=-1), "... d two -> ...(d two)", two=2)\\
         $$
     - 当输入rotate时，旋转矩阵的生成方式为：
         $$
         rotate(x) = x \cdot rotate\\
         $$

-   等价计算逻辑：
    
     可使用`fused_rotary_position_embedding`等价替换`torch_npu.npu_rotary_mul`，两者计算逻辑一致。
     
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

## 函数原型

```
torch_npu.npu_rotary_mul(input, r1, r2, rotary_mode='half', rotate=None) -> Tensor
```

> [!NOTE]  
> 在模型训练场景中，正向算子的输入`input`将被保留以供反向计算时使用。在`r1`、`r2`不需要计算反向梯度场景下（`requires_grad=False`），使用该接口相较融合前小算子使用的设备内存占用会有所增加。输入`rotate`不支持反向传播。

## 参数说明

- **input** (`Tensor`)：必选参数，输入维度支持3维、4维，数据类型支持`float16`，`bfloat16`，`float32`。
- **r1** (`Tensor`)：必选参数，表示$cos$旋转系数，输入维度支持3维、4维，数据类型支持`float16`，`bfloat16`，`float32`。
- **r2** (`Tensor`)：必选参数，表示$sin$旋转系数，输入维度支持3维、4维，数据类型支持`float16`，`bfloat16`，`float32`。
- **rotary_mode** (`str`)：可选参数，数据类型支持`str`，用于选择计算模式，支持`half`、`interleave`两种模式。默认值为`half`。
- **rotate** (`Tensor`)：可选参数，表示实现input位置变换的等价变化矩阵，输入维度支持2维，数据类型支持`float16`，`bfloat16`，`float32`，构造方式参考调用示例，默认为空。

## 返回值说明
`Tensor`

输出计算结果，shape和dtype需与`input`一致。

## 约束说明

- **`jit_compile=False`场景**（适用<term>Atlas A2 训练系列产品</term>，<term>Atlas A3 训练系列产品</term>）：
    - half模式：

        `input`：layout支持：$BNSD、BSND、SBND、TND；D < 896$，且为2的倍数；$B, N < 1000$；当需要计算$cos/sin$的反向梯度时，$B*N <= 1024$。

        `r1、r2`：数据范围：[-1, 1]；对应`input` layout的支持情况：

        - x为$BNSD$: $11SD、B1SD、BNSD$；
        - x为$BSND$: $1S1D、BS1D、BSND$；
        - x为$SBND$: $S11D、SB1D、SBND$;
        - x为$TND$: $T1D、TND$。

            > [!NOTICE]  
            > half模式下，当输入layout是$BNSD$，且$D$为非32Bytes对齐时，建议不使用该融合算子（模型启动脚本中不开启`--use-fused-rotary-pos-emb`选项），否则可能出现性能下降。

    - interleave模式：

        `input`：layout支持：$BNSD、BSND、SBND、TND; B*N < 1000; D < 896$, 且$D$为2的倍数;

        `r1、r2`：数据范围：[-1, 1]；对应`input` layout的支持情况：

        - x为$BNSD: 11SD$;
        - x为$BSND: 1S1D$;
        - x为$SBND: S11D$；
        - x为$TND: T1D$。

- **`jit_compile=True`场景**（适用<term>Atlas 训练系列产品</term>，<term>Atlas A2 训练系列产品</term>，<term>Atlas 推理系列产品</term>）：

     仅支持`rotary_mode`为half模式，且`r1/r2` layout一般为$11SD、1S1D、S11D$。

     shape要求输入为4维，其中$B$维度和$N$维度数值需小于等于1000，$D$维度数值为128。

     广播场景下，广播轴的总数据量不能超过1024。

- rotate推荐使用场景
  - interleave模式
  - half模式仅在以下场景时推荐使用：输入矩阵x需要在最后一个维度切分多份时，可以通过构造旋转编码矩阵实现一次调用获得性能收益，以x的layout为BSND需要切分为3份为例：
     x切分为3份，$x = [x1|x2|x3]_{(dim=4)} ∈ R^{B×S×N×D}, x1 ∈ R^{B×S×N×D1},x2 ∈ R^{B×S×N×D2},x3 ∈ R^{B×S×N×D3}, 其中D = D1 + D2 + D3$，
     那么可以构造一个rotate矩阵，实现调用一次完成x的旋转位置编码计算功能，rotate矩阵构造如下：
     $$rotate = diag(rotate1, rotate2, rotate3) = \begin{pmatrix}rotate1&0&0\\0&rotate2&0\\0&0&rotate3\\\end{pmatrix}$$
     其中rotate1、rotate2、rotate3分别为x1、x2、x3的旋转编码矩阵，单个旋转矩阵构建参考调用示例。
## 调用示例

- 四维输入示例：
    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.rand(2, 2, 5, 128).npu()
    >>> r1 = torch.rand(1, 2, 1, 128).npu()
    >>> r2 = torch.rand(1, 2, 1, 128).npu()
    >>> out = torch_npu.npu_rotary_mul(x, r1, r2)
    >>> out.shape
    torch.Size([2, 2, 5, 128])
    >>> out
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
- 三维输入示例：
    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.rand(2, 5, 128).npu()
    >>> r1 = torch.rand(2, 1, 128).npu()
    >>> r2 = torch.rand(2, 1, 128).npu()
    >>> out = torch_npu.npu_rotary_mul(x, r1, r2，"half")
    >>> out
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

- rotate生成示例：
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
        out = torch_npu.npu_rotary_mul(x, r1, r2，"interleave", inter_mat_128.npu())
    ```