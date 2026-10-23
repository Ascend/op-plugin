# torch_npu.npu_grouped_matmul_add

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- API function: Implements grouped matrix multiplication and accumulation (GroupedMatmulAdd). The dimensions of each group can be different. The basic function is matrix multiplication plus addition, such as $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i]+y_i[m_i,n_i], i=1...g$, where g is the number of groups and $m_i/k_i/n_i$ are the corresponding shapes. The input and output data types are all Tensors, and the K axis is grouped.
  - K-axis grouping: $k_i$ differs for each group, but $m_i/n_i$ are the same for each group.
  - `group_list` supports two interpretation methods (specified by `group_list_type`):
    - `group_list_type` is 0 (default): The values in `group_list` are the cumulative sum (cumsum) of the group axis sizes.
    - `group_list_type` is 1: The values in `group_list` are the size of each group on the group axis.
- Formula:

  $$
  yRef_i^{out}=x_i\times weight_i + yRef_i^{in}
  $$

  Here, $yRef$ corresponds to the input parameter `self`, and the output is the accumulation of `self` and the grouped matrix multiplication result.

> [!NOTE]
>
> This API provides an in-place version, `torch_npu.npu_grouped_matmul_add_`, which has the same parameters and return value, directly accumulates on the input `self`, and returns `self` itself.

## Prototype

```python
torch_npu.npu_grouped_matmul_add(self, x, weight, group_list, *, transpose_x=True, transpose_weight=False,
                                 group_type=2, group_list_type=0) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. Input matrix for in-place accumulation, $yRef$ in the formula. The shape is [g, M, N], where g is the number of groups. The data type can be `torch.float32`. The data layout can be ND.
- **`x`** (`Tensor`): Required. Input $x$ in the formula. The number of dimensions can be 2. The data layout can be ND. The data type can be `torch.float16` or `torch.bfloat16` and must be the same as that of `weight`. Non-contiguous Tensors are supported. `x` must be transposed, so `transpose_x` supports only True.
- **`weight`** (`Tensor`): Required. Weight, $weight$ in the formula. The number of dimensions can be 2. The data layout can be ND. The data type can be `torch.float16` or `torch.bfloat16` and must be the same as that of `x`. Non-contiguous Tensors are supported. `weight` is not transposed, so `transpose_weight` supports only False.
- **`group_list`** (`Tensor`): Required. Distribution of matmul sizes along the input and output group axis. The data type can be `torch.int64`. The data layout can be ND. The first dimension supports a maximum of 1024 groups.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`transpose_x`** (`bool`): Optional. Whether the `x` matrix is transposed. Currently, only True is supported. The default value is True.
- **`transpose_weight`** (`bool`): Optional. Whether the `weight` matrix is transposed. Currently, only False is supported. The default value is False.
- **`group_type`** (`int`): Optional. Group type. Currently, only 2 is supported (K-axis grouping), that is, the K axis ($k_i$) of $x_i$ and $weight_i$ is split by group, and $m_i/n_i$ are the same for each group. The default value is 2.
- **`group_list_type`** (`int`): Optional. Interpretation method of the values in `group_list`. Currently, only two values are supported. The default value is 0.
  - 0: The values in `group_list` are the cumulative sum (cumsum) of the group axis sizes. `group_list` must be a non-negative monotonically non-decreasing sequence, and the last value must not be greater than the first dimension of `x`. For example, when K=256 and E=4 (group sizes are 64, 0, 128, and 64): `[64, 64, 192, 256]`.
  - 1: The values in `group_list` are the size of each group on the group axis. `group_list` must be a non-negative sequence, and the sum of its values must not be greater than the first dimension of `x`. For example, when K=256 and E=4: `[64, 0, 128, 64]`.

## Return Values

**`output`** (`Tensor`): Result of grouped matrix multiplication and accumulation, $yRef^{out}$ in the formula. The shape is [g, M, N], which is the same as that of the input `self`. The data type is `torch.float32`.

## Constraints

- This API supports training and inference scenarios.
- This API supports single-operator mode and graph mode.
- Deterministic computation: The deterministic implementation is used by default.
- Supported input type combinations:
  - `x` is `torch.float16`, `weight` is `torch.float16`, and `self` is `torch.float32`.
  - `x` is `torch.bfloat16`, `weight` is `torch.bfloat16`, and `self` is `torch.float32`.
- The data types of `x` and `weight` must be the same.
- Each dimension of each group Tensor in `x` and `weight`, after 32-byte alignment, must be less than the maximum value of `torch.int32`, 2147483647.
- The last dimension of each group Tensor in `x` and `weight` must be less than 65536. The last dimension of `x` refers to the M axis after transposition, and the last dimension of `weight` refers to the N axis when not transposed.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    
    torch.manual_seed(0)
    # The shape of x is [512, 256]. After transposition, M=256 and K=512. group_list splits the K axis into [0:256] and [256:512].
    x = torch.randn(512, 256, dtype=torch.bfloat16).npu()
    weight = torch.randn(512, 256, dtype=torch.bfloat16).npu()
    self_t = torch.zeros(2, 256, 256, dtype=torch.float32).npu()
    group_list = torch.tensor([256, 512], dtype=torch.int64).npu()
    out = torch_npu.npu_grouped_matmul_add(self_t, x, weight, group_list, transpose_x=True)
    print(out.shape, out.dtype)
    print(out[0, 0, :6])
    print(out[1, 0, :4])
    ```

    Output:

    ```text
    torch.Size([2, 256, 256]), torch.float32
    tensor([-14.8446,  -4.4171,   7.6527,  10.9340, -22.6788, -15.8647], device='npu:0')
    tensor([5.2137, 8.1306, 1.2067, 2.9381], device='npu:0')
    ```

- Single-operator call with non-zero self (accumulation semantics)

    ```python
    import torch
    import torch_npu
    
    torch.manual_seed(0)
    x = torch.randn(512, 256, dtype=torch.bfloat16).npu()
    weight = torch.randn(512, 256, dtype=torch.bfloat16).npu()
    self_t = torch.full((2, 256, 256), 0.5, dtype=torch.float32).npu()
    group_list = torch.tensor([256, 512], dtype=torch.int64).npu()
    out = torch_npu.npu_grouped_matmul_add(self_t, x, weight, group_list, transpose_x=True)
    print(out[0, 0, :3]) 
    ```

    Output:

    ```text
    tensor([-14.3446,  -3.9171,   8.1527], device='npu:0')
    ```

- Single-operator call, in-place version

    ```python
    import torch
    import torch_npu
    
    torch.manual_seed(0)
    x = torch.randn(512, 256, dtype=torch.bfloat16).npu()
    weight = torch.randn(512, 256, dtype=torch.bfloat16).npu()
    self_t = torch.zeros(2, 256, 256, dtype=torch.float32).npu()
    group_list = torch.tensor([256, 512], dtype=torch.int64).npu()
    out = torch_npu.npu_grouped_matmul_add_(self_t, x, weight, group_list, transpose_x=True)
    print(out is self_t)  # True
    print(out[0, 0, :3]) # tensor([-14.8446,  -4.4171,   7.6527], device='npu:0')
    ```

    Output:

    ```text
    True
    tensor([-14.8446,  -4.4171,   7.6527], device='npu:0')
    ```
