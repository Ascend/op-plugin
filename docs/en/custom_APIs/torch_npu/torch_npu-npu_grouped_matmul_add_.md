# torch_npu.npu_grouped_matmul_add_

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- **API function**: In micro-batch training scenarios, micro-batch gradient accumulation involves many fusion scenarios in which GroupedMatmul operations are followed by InplaceAdd operations. This operator (GroupedMatmulAdd) fuses these operators in non-quantized scenarios to improve network performance.

- **Formula**:

  $$
  self = self + x @ weight
  $$

## Prototype

```python
torch_npu.npu_grouped_matmul_add_(self, x, weight, group_list, *, transpose_x=True, transpose_weight=False, group_type=2, group_list_type=0) -> torch.Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. Matrix to be accumulated. The data type can be `torch.float32`. The Tensor can be 3D with shape (g, M, N). The data layout can be ND.
- **`x`** (`Tensor`): Required. Left matrix in the matrix multiplication. The data type can be `torch.float16` or `torch.bfloat16`. The Tensor can be 2D with shape (K, M). The data layout can be ND.
- **`weight`** (`Tensor`): Required. Right matrix in the matrix multiplication. The data type can be `torch.float16` or `torch.bfloat16`. The Tensor can be 2D with shape (K, N). The data layout can be ND.
- **`group_list`** (`Tensor`): Required. Distribution of matmul sizes along the input and output group axis. The data type can be `torch.int64`. The Tensor is 1D with shape (g,). The data layout can be ND.
  - When `group_list_type` is 0, `group_list` must be a non-negative monotonically non-decreasing sequence, and the last value must not be greater than the first dimension of `x`.
  - When `group_list_type` is 1, `group_list` must be a non-negative sequence, and the sum of its values must not be greater than the first dimension of `x`.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`transpose_x`** (`bool`): Optional. Indicates whether the `x` matrix is transposed. True indicates that `x` is transposed (default), and False indicates that `x` is not transposed. Currently, only True is supported.
- **`transpose_weight`** (`bool`): Optional. Indicates whether the `weight` matrix is transposed. False indicates that `weight` is not transposed (currently the only supported value), and True indicates that `weight` is transposed.
- **`group_type`** (`int`): Optional. Indicates the axis to be grouped. The data type can be `int64`. Only 2 (default) is supported.
- **`group_list_type`** (`int`): Optional. Indicates the expression form of `group_list`. The data type can be `int64`.
  - 0 (default): The values in `group_list` are the cumulative sum (cumsum) of the group axis sizes.
  - 1: The values in `group_list` are the size of each group on the group axis.

## Return Values

`Tensor`

Returns the `self` Tensor itself, which is the final result matrix obtained by adding the result of the groupedMatmul computation to the matrix to be accumulated. The supported data type, shape, and data layout are the same as those of the input `self`.

## Constraints

- This API is supported only in training scenarios.
- This API supports only single-operator mode.
- Variables used in the shape description:
  - g: Number of groups, in the range 1 to 1024.
  - Each dimension of the `x` and `weight` matrices, after 32-byte alignment, must be less than the maximum value of `torch.int32`, 2147483647.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    M = 576
    K = 512
    N = 7168
    g = 4
    y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu()
    x1 = torch.randint(-1, 1, (K, M), dtype=torch.float16).npu()
    x2 = torch.randint(-1, 1, (K, N), dtype=torch.float16).npu()
    group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu()
    y = torch_npu.npu_grouped_matmul_add_(y, x1, x2, group_list, transpose_x=True, transpose_weight=False, group_type=2, group_list_type=0)
    print(y)
    ```
