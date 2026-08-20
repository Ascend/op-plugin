# torch\_npu.npu\_add\_quant\_matmul\_

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## Function

- Description:

    In micro-batch training scenarios requiring gradient accumulation across micro-batches, numerous cases exist where a `QuantBatchMatmul` operation is followed by an `InplaceAdd` operation. This operator (`QuantBatchMatmulInplaceAdd`) fuses these operations to improve network performance.

- Formulas:

    The formula for the MX quantization scenario is as follows. For more information about quantization techniques, see "Basic Concepts > Introduction to Quantization Mode" in [CANN Operator Library](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/common/quant_mode_introduction.md).

    ![](../../figures/en-us_formulaimage_0000002521244910.png)

    `gsk` represents the block size for MX quantization along the K axis, which is 32. `x1_slice<sub>i</sub>` represents a vector of length `gsk` in row `m` of `x1<sub>i</sub>`, and `x2_slice<sub>i</sub>` represents a vector of length `gsk` in column `n` of `x2<sub>i</sub>`. Slicing along the K axis starts at `j * gsk`. The value range of `j` is [0, `k_loops`), and `k_loops = ceil(K<sub>i</sub> / gsk)`. The length of the final slice can be less than `gsk`.

    The computation formula for the T-T quantization scenario is as follows:

    ![](../../figures/en-us_formulaimage_0000002594063675.png)

## Prototype

```python
torch_npu.npu_add_quant_matmul_(self, x1, x2, x2_scale, *, x1_scale=None, group_sizes=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None) -> torch.Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. Matrix to which accumulation is applied. The data type can be `float32`. This parameter must be a 2D tensor with shape `[M, N]`. The data layout can be `ND`.
- **`x1`** (`Tensor`): Required. Left matrix in matrix multiplication. The data type can be `float8_e5m2`, `float8_e4m3fn`, or `hifloat8`. This parameter must be a 2D tensor with shape `[K, M]`. The data layout can be `ND`.
- **`x2`** (`Tensor`): Required. Right matrix in matrix multiplication. The data type can be `float8_e5m2`, `float8_e4m3fn`, or `hifloat8`. This parameter must be a 2D tensor with shape `[K, N]`. The data layout can be `ND`.
- **`x2_scale`** (`Tensor`): Required. Scaling factor for the right matrix in matrix multiplication. The data type can be `float8_e8m0fnu` or `float32`. This parameter must be a 3D tensor. When the data type is `float8_e8m0fnu`, the optional parameter `x2_scale_dtype` must be configured to the corresponding type. In this case, the dtype of `x2_scale` itself is ignored, but its underlying type must remain an 8-bit data type to ensure the correct shape. The data layout can be `ND`.
- **`*`**: Position delimiter. Variables before this delimiter are position-dependent and must be passed in order. Variables after this delimiter are optional keyword arguments and must be assigned using key-value pairs. If not specified, their default values are used.
- **`x1_scale`** (`Tensor`): Optional. Scaling factor for the left matrix in matrix multiplication. The data type can be `float8_e8m0fnu` or `float32`. This parameter must be a 3D tensor. When the data type is `float8_e8m0fnu`, the optional parameter `x1_scale_dtype` must be configured to the corresponding type. In this case, the dtype of `x1_scale` itself is ignored, but its underlying type must remain an 8-bit data type to ensure the correct shape. The data layout can be `ND`.
- **`group_sizes`** (`List[int]`): Optional. Default value is `None`.
  - When set to a non-`None` value, only a 3-element list in the form `[group_m, group_n, group_k]` is supported, representing the quantization grouping along the `m`, `n`, and `k` dimensions, respectively. For example, `group_m` indicates that every `group_m` elements along the `m` dimension correspond to one quantization parameter.
  - When one or more values in `[group_m, group_n, group_k]` are `0`, the API automatically adjusts those values based on the input shapes of `x1`, `x2`, `x1_scale`, and `x2_scale`. For example, if `group_m = 0`, the grouping value along the `m` dimension is inferred using the formula `group_m = m / scale_m`, where `m` must be divisible by `scale_m`. Here, `m` is the `m` dimension in the shape of `x1`, and `scale_m` is the `m` dimension in the shape of `x1_scale`.
  - For MX quantization, the only supported value is `[1, 1, 32]`. For T-T quantization, the only supported value is `[0, 0, 0]`.

- **`x1_scale_dtype`** (`int`): Optional. Explicitly specifies the data type of `x1_scale` when it cannot be represented using native Torch data types. The default value is `None`, indicating that the actual data type is the same as the dtype of `x1_scale`. Currently, the data type can only be `float8_e8m0fnu`.

- **`x2_scale_dtype`** (`int`): Optional. Explicitly specifies the data type of `x2_scale` when it cannot be represented using native Torch data types. The default value is `None`, indicating that the actual data type is the same as the dtype of `x2_scale`. Currently, the data type can only be `float8_e8m0fnu`.

## Return Values

**`self`** (`Tensor`): Final result matrix obtained by adding the output matrix of the `QuantBatchMatmul` computation to the accumulation matrix. The data type, shape, and data layout are identical to those of the input `self`.

## Constraints

- This API can be used in training scenarios.
- This API supports single-operator mode and TorchAir graph mode.
- Data type constraints:

    | Scenario | x1 | x2 | x2_scale | x1_scale | self |
    | --- | --- | --- | --- | --- | --- |
    | MX quantization | <code>float8_e4m3fn</code>/<code>float8_e5m2</code> | <code>float8_e4m3fn</code>/<code>float8_e5m2</code> | <code>float8_e8m0fnu</code> | <code>float8_e8m0fnu</code> | <code>float32</code> |
    | T-T quantization | <code>hifloat8</code> | <code>hifloat8</code> | <code>float32</code> | <code>float32</code> | <code>float32</code> |

- Shape constraints:

    | Scenario | x1 | x2 | x2_scale | x1_scale | self |
    | --- | --- | --- | --- | --- | --- |
    | MX quantization | <code>[K, M]</code> | <code>[K, N]</code> | <code>[ceil(K / 64), N, 2]</code> | <code>[ceil(K / 64), M, 2]</code> | <code>[M, N]</code> |
    | T-T quantization | <code>[K, M]</code> | <code>[K, N]</code> | <code>[1]</code> | <code>[1]</code> | <code>[M, N]</code> |

## Examples

- Single-operator call
  - Single-operator call for MX quantization

    ```python
    import math
    import torch
    import torch_npu
    M = 576
    N = 7168
    K = 512
    y = torch.randint(-1, 1, (M, N), dtype=torch.float32).npu()
    x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(0,1)
    x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    x2_scale = torch.randint(-1, 1, (math.ceil(K/64), N, 2), dtype=torch.int8).npu()
    x1_scale = torch.randint(-1, 1, (math.ceil(K/64), M, 2), dtype=torch.int8).npu().transpose(0,1)
    y = torch_npu.npu_add_quant_matmul_(y, x1, x2, x2_scale,x1_scale = x1_scale, x1_scale_dtype=torch_npu.float8_e8m0fnu, x2_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes = [1,1,32])
    ```

  - Single-operator call for T-T quantization

    ```python
    import math
    import torch
    import torch_npu
    M = 16
    N = 16
    K = 16
    y = torch.randint(-1, 1, (M, N), dtype=torch.float32).npu()
    x1 = torch.randint(0, 1, (K, M), dtype=torch.uint8).npu().transpose(0,1)
    x2 = torch.randint(0, 1, (K, N), dtype=torch.uint8).npu()
    x2_scale = torch.randint(-1, 1, (1,), dtype=torch.float32).npu()
    x1_scale = torch.randint(-1, 1, (1,), dtype=torch.float32).npu()
    y = torch_npu.npu_add_quant_matmul_(y, x1, x2, x2_scale,x1_scale = x1_scale,x1_dtype =  torch_npu.hifloat8, x2_dtype = torch_npu.hifloat8, x1_scale_dtype=None, x2_scale_dtype=None, group_sizes = [0,0,0])
    ```

- Graph mode call
  - Graph mode call for MX quantization

    ```python
    import math
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
    N = 7168
    K = 512
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, y, x1, x2, x2_scale, x1_scale, x1_scale_dtype, x2_scale_dtype):
            return torch_npu.npu_add_quant_matmul_(y, x1.transpose(0,1), x2, x2_scale, x1_scale = x1_scale.transpose(0, 1), x1_scale_dtype=x1_scale_dtype, x2_scale_dtype=x2_scale_dtype)
    def main():
        y = torch.randint(-1, 1, (M, N), dtype=torch.float32).npu()
        x1 = torch.randint(-1, 1, (K, M), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        x2 = torch.randint(-1, 1, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        x2_scale = torch.randint(-1, 1, (math.ceil(K/64),N, 2), dtype=torch.int8).npu()
        x1_scale = torch.randint(-1, 1, (math.ceil(K/64),M, 2), dtype=torch.int8).npu()
        model = Model().npu()
        model = torch.compile(model, backend=npu_backend)
        y = model(y, x1, x2, x2_scale, x1_scale, torch_npu.float8_e8m0fnu, torch_npu.float8_e8m0fnu)
        print(y.cpu())
     
    if __name__ == '__main__':
        main()
    ```

  - Graph mode call for T-T quantization

    ```python
    import math
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import os
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    #os.environ["ENABLE_ACLNN"] = "true"
    M = 16
    N = 16
    K = 16
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, y, x1, x2, x2_scale, x1_scale, x1_scale_dtype, x2_scale_dtype):
            return torch_npu.npu_add_quant_matmul_(y, x1.transpose(0,1), x2, x2_scale, x1_dtype =  torch_npu.hifloat8, x2_dtype = torch_npu.hifloat8, x1_scale = x1_scale, x1_scale_dtype=None, x2_scale_dtype=None)
    def main():
        y = torch.randint(-1, 1, (M, N), dtype=torch.float32).npu()
        x1 = torch.randint(0, 1, (K, M), dtype=torch.uint8).npu()
        x2 = torch.randint(0, 1, (K, N), dtype=torch.uint8).npu()
        x2_scale = torch.randint(-1, 1, (1,), dtype=torch.float32).npu()
        x1_scale = torch.randint(-1, 1, (1,), dtype=torch.float32).npu()
        model = Model().npu()
        model = torch.compile(model, backend=npu_backend)
        y = model(y, x1, x2, x2_scale, x1_scale, None, None)
        print(y.cpu())
    
    if __name__ == '__main__':
        main()
    ```
