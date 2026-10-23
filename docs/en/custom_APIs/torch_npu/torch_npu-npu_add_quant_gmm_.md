# torch_npu.npu_add_quant_gmm_

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- **API function**:

  In micro-batch training scenarios, micro-batch gradient accumulation requires a large number of fusion scenarios in which GroupedMatmul operations are followed by InplaceAdd operations. This operator (QuantGroupedMatmulInplaceAdd) fuses these operators to improve network performance.

- **Formulas**:

  The formulas for different quantization scenarios are as follows. For more information about quantization, see [Quantization](https://gitcode.com/cann/ops-math/blob/9.2.0/docs/en/context/quant_mode_introduction.md) in the *CANN Operator Library*.

  - T-T/T-C quantization scenario:

    $$
    \mathit{self} = \mathit{self} + \mathit{x1} \times \mathit{x2} * \mathit{x2\_scale} * \mathit{x1\_scale}
    $$

  - MX quantization scenario:

    $$
    \begin{aligned}
    \mathit{self}[m,n] = {} &\mathit{self}[m,n] + \\
    &\sum_{j=0}^{k\_loops-1}\left(\left(\sum_{k=0}^{gsk-1}(\mathit{x1\_slice}_i \times \mathit{x2\_slice}_i)\right) \times \left(\mathit{x1\_scale}_i[m,j] \times \mathit{x2\_scale}_i[j,n]\right)\right)
    \end{aligned}
    $$

    Here, gsk is the MX quantization block size on the K axis, which is 32. K<sub>i</sub> is the K size of each group. x1\_slice<sub>i</sub> is the gsk-length vector in row m of x1<sub>i</sub>, and x2\_slice<sub>i</sub> is the gsk-length vector in column n of x2<sub>i</sub>. The K axis starts from j\*gsk. The value range of j is [0, k\_loops), and k\_loops=ceil(K<sub>i</sub>/gsk). The last slice may be shorter than gsk.

## Prototype

```python
torch_npu.npu_add_quant_gmm_(self, x1, x2, x2_scale, group_list, *, x1_scale=None, group_list_type=0, group_sizes=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None)
```

## Parameters

> Variables used in the shape description:
>
> - g: Number of groups. The value range is 1 to 1024.
> - M: Size of the second-to-last dimension of the output matrix. The value range is 1 to 2147483647.
> - N: Size of the last dimension of the output matrix. The value range is 1 to 2147483647.
> - K: Size of the reduce axis of the matrix multiplication. The value range is 1 to 2147483647.
> Note: When M, N, or K is the innermost axis of the input matrix, the value must be less than 2097152.

- **`self`** (`Tensor`): Required. Matrix to be accumulated. The data type can be `torch.float32`. The Tensor can be 3D with shape `(g, M, N)`. The data layout can be ND.
- **`x1`** (`Tensor`): Required. Left matrix in the matrix multiplication. The data type can be `torch.float8_e5m2`, `torch.float8_e4m3fn`, or `torch_npu.hifloat8`. The Tensor can be 2D with shape `(K, M)`. For `torch_npu.hifloat8`, the optional parameter `x1_dtype` must be set to the corresponding type. In this case, the `dtype` of `x1` itself does not take effect, but the `dtype` of `x1` itself must still be an 8-bit data type to ensure the correct shape. The data layout can be ND.
- **`x2`** (`Tensor`): Required. Right matrix in the matrix multiplication. The data type can be `torch.float8_e5m2`, `torch.float8_e4m3fn`, or `torch_npu.hifloat8`. The Tensor can be 2D with shape `(K, N)`. For `torch_npu.hifloat8`, the optional parameter `x2_dtype` must be set to the corresponding type. In this case, the `dtype` of `x2` itself does not take effect, but the `dtype` of `x2` itself must still be an 8-bit data type to ensure the correct shape. The data layout can be ND.
- **`x2_scale`** (`Tensor`): Required. Scaling factor of the right matrix in the matrix multiplication. The data type can be `torch.float8_e8m0fnu` or `torch.float32`. The shape can have 1 to 3 dimensions. For `torch.float8_e8m0fnu`, the optional parameter `x2_scale_dtype` must be set to the corresponding type. In this case, the `dtype` of `x2_scale` itself does not take effect, but the `dtype` of `x2_scale` itself must still be an 8-bit data type to ensure the correct shape. The data layout can be ND.
- **`group_list`** (`Tensor`): Required. Distribution of matmul sizes along the input and output group axis. The data type can be `torch.int64`. The Tensor is 1D with shape `(g,)`. The data layout can be ND. When `group_list_type` is 0, `group_list` must be a non-negative monotonically non-decreasing sequence, and the last value must not be greater than the first dimension of the input Tensor. When `group_list_type` is 1, `group_list` must be a non-negative sequence, and the sum of the values must not be greater than the first dimension of the input Tensor. The values in `group_list` constrain the valid part of the output data. The part not specified in `group_list` does not participate in the update.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`x1_scale`** (`Tensor`): Optional. Scaling factor of the left matrix in the matrix multiplication. The data type can be `torch.float8_e8m0fnu` or `torch.float32`. The shape can have 1, 2, or 3 dimensions. For `torch.float8_e8m0fnu`, the optional parameter `x1_scale_dtype` must be set to the corresponding type. In this case, the `dtype` of `x1_scale` itself does not take effect, but the `dtype` of `x1_scale` itself must still be an 8-bit data type to ensure the correct shape. The data layout can be ND.
- **`group_list_type`** (`int`): Optional. Expression form of `group_list`. The data type can be `int32`.

  - 0: Default value. The values in `group_list` are the cumulative sum (cumsum) of the group axis sizes.
  - 1: The values in `group_list` are the size of each group on the group axis.

- **`group_sizes`** (`List[int]`): Optional. The list size is 3, indicating how many parameters on the M, N, and K axes share one scaling factor. The data type can be `torch.int32`. Currently, only the default value None is supported.
- **`x1_dtype`** (`int`): Optional. Used to explicitly specify the data type of `x1` when `x1` cannot be represented by a native torch data type.

  - None: Default value. The actual data type of the input `x1` is the same as the `dtype` of the input `x1`.
  - The actual data type of the input `x1`. Currently, only `torch_npu.hifloat8` is supported.

- **`x2_dtype`** (`int`): Optional. Used to explicitly specify the data type of `x2` when `x2` cannot be represented by a native torch data type.

  - None: Default value. The actual data type of the input `x2` is the same as the `dtype` of the input `x2`.
  - The actual data type of the input `x2`. Currently, only `torch_npu.hifloat8` is supported.

- **`x1_scale_dtype`** (`int`): Optional. Used to explicitly specify the data type of `x1_scale` when `x1_scale` cannot be represented by a native torch data type.

  - None: Default value. The actual data type of the input `x1_scale` is the same as the `dtype` of the input `x1_scale`.
  - The actual data type of the input `x1_scale`. Currently, only `torch.float8_e8m0fnu` is supported.

- **`x2_scale_dtype`** (`int`): Optional. Used to explicitly specify the data type of `x2_scale` when `x2_scale` cannot be represented by a native torch data type.

  - None: Default value. The actual data type of the input `x2_scale` is the same as the `dtype` of the input `x2_scale`.
  - The actual data type of the input `x2_scale`. Currently, only `torch.float8_e8m0fnu` is supported.

## Return Values

**`self`** (`Tensor`): Final result matrix obtained by adding the result of the groupedMatmul computation to the matrix to be accumulated. The data type, shape, and data layout are the same as those of the input `self`.

## Constraints

- This API supports training scenarios.
- This API supports single-operator mode and TorchAir graph mode.
- The first dimension of `group_list` supports a maximum of 1024 groups.
- Data type constraints:

  | Scenario | x1 | x2 | x2_scale | x1_scale | self |
  | --- | --- | --- | --- | --- | --- |
  | T-T/T-C quantization | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | `torch.float32` |
  | MX quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float32` |

- Shape constraints

  | Scenario | x2_scale | x1_scale |
  | --- | --- | --- |
  | T-T/T-C quantization | pertensor scenario: 2D or 1D Tensor with shape `(g, 1)` or `(g,)`; perchannel scenario: 2D Tensor with shape `(g, N)` | 2D or 1D Tensor with shape `(g, 1)` or `(g,)` |
  | MX quantization | 3D Tensor with shape `((K / 64) + g, N, 2)` | 3D Tensor with shape `((K / 64) + g, M, 2)` |

## Examples

- Single-operator call
  - T-C quantization scenario

    ```python
    import torch
    import torch_npu
    M = 576
    K = 512
    N = 7168
    g = 4
    y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu()
    x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).npu().transpose(0,1)
    x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).npu()
    x2_scale = torch.randint(-1, 1, (g, N), dtype=torch.float32).npu()
    x1_scale = torch.randint(-1, 1, (g,), dtype=torch.float32).npu()
    group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu()
    y = torch_npu.npu_add_quant_gmm_(y, x1, x2, x2_scale, group_list, x1_scale = x1_scale, group_list_type=0, x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8)
    print(y.cpu())
    ```

  - MX quantization

    ```python
    import torch
    import torch_npu
    M = 576
    K = 512
    N = 7168
    g = 4
    y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu()
    x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(0,1)
    x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    x2_scale = torch.randint(-1, 1, (int(K/64 + g), N, 2), dtype=torch.int8).npu()
    x1_scale = torch.randint(-1, 1, (int(K/64 + g), M, 2), dtype=torch.int8).npu().transpose(0,1)
    group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu()
    y = torch_npu.npu_add_quant_gmm_(y, x1, x2, x2_scale, group_list, x1_scale = x1_scale, group_list_type=0, x1_scale_dtype=torch_npu.float8_e8m0fnu, x2_scale_dtype=torch_npu.float8_e8m0fnu)
    print(y.cpu())
    ```

- Graph mode call
  - T-C quantization scenario

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import os

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    #os.environ["ENABLE_ACLNN"] = "true"
    M = 576
    K = 512
    N = 7168
    g = 4
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, y, x1, x2, x2_scale, group_list, x1_scale, group_list_type, x1_dtype, x2_dtype):
            return torch_npu.npu_add_quant_gmm_(y, x1.transpose(0,1), x2, x2_scale, group_list, x1_scale = x1_scale, group_list_type=group_list_type, x1_dtype=x1_dtype, x2_dtype=x2_dtype)

    def main():
        y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu()
        x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).npu()
        x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).npu()
        x2_scale = torch.randint(-1, 1, (g, N), dtype=torch.float32).npu()
        x1_scale = torch.randint(-1, 1, (g,), dtype=torch.float32).npu()
        group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu()
        group_list_type = 0
        model = Model().npu()
        model = torch.compile(model, backend=npu_backend)
        y = model(y, x1, x2, x2_scale, group_list, x1_scale, group_list_type, torch_npu.hifloat8, torch_npu.hifloat8)
        print(y.cpu())

    if __name__ == '__main__':
        main()
    ```

  - MX quantization

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import os

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    #os.environ["ENABLE_ACLNN"] = "true"
    M = 576
    K = 512
    N = 7168
    g = 4
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, y, x1, x2, x2_scale, group_list, x1_scale, group_list_type, x1_scale_dtype, x2_scale_dtype):
            return torch_npu.npu_add_quant_gmm_(y, x1.transpose(0,1), x2, x2_scale, group_list, x1_scale = x1_scale.transpose(0, 1), group_list_type=group_list_type, x1_scale_dtype=x1_scale_dtype, x2_scale_dtype=x2_scale_dtype)

    def main():
        y = torch.randint(-1, 1, (g, M, N), dtype=torch.float32).npu()
        x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        x2_scale = torch.randint(-1, 1, (int(K/64 + g), N, 2), dtype=torch.int8).npu()
        x1_scale = torch.randint(-1, 1, (int(K/64 + g), M, 2), dtype=torch.int8).npu()
        group_list = torch.Tensor([8, 181, 415, 512]).to(torch.int64).npu()
        group_list_type = 0
        model = Model().npu()
        model = torch.compile(model, backend=npu_backend)
        y = model(y, x1, x2, x2_scale, group_list, x1_scale, group_list_type, torch_npu.float8_e8m0fnu, torch_npu.float8_e8m0fnu)
        print(y.cpu())

    if __name__ == '__main__':
        main()
    ```
