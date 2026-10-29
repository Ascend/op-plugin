# torch\_npu.npu\_add\_quant\_matmul\_

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- Function:

    In micro-batch training scenarios, micro-batch gradient accumulation involves a large number of scenarios in which QuantBatchMatmul operations are followed by InplaceAdd operations. This operator (QuantBatchMatmulInplaceAdd) fuses the preceding operations to improve network performance.

- Formulas:

    The formula for the mx quantization scenario is as follows. For more information about quantization technologies, see [Quantization Introduction](https://gitcode.com/cann/ops-math/blob/9.2.0/docs/en/context/quant_mode_introduction.md) in the *CANN Operator Library*.

    ![](../../figures/en-us_formulaimage_0000002521244910.png)

    In the formula, gsk indicates the block size of the mx quantization on the K axis, that is, 32. x1\_slice<sub>i</sub> indicates the vector of the m-th row of x1<sub>i</sub> with a length of gsk, and x2\_slice<sub>i</sub> indicates the vector of the n-th column of x2<sub>i</sub> with a length of gsk. The K axis is always sliced from j\*gsk, and the value range of j is [0, k\_loops), where k\_loops=ceil(K<sub>i</sub>/gsk). A final slice shorter than gsk is supported.

    The formula for the T-T quantization scenario is as follows.

    ![](../../figures/en-us_formulaimage_0000002594063675.png)

## Prototype

```python
torch_npu.npu_add_quant_matmul_(self, x1, x2, x2_scale, *, x1_scale=None, group_sizes=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None) -> torch.Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. Matrix to which accumulation is applied. The data type can be `torch.float32`. This parameter must be a 2D tensor with shape `[M, N]`. The data layout can be `ND`.
- **`x1`** (`Tensor`): Required. Left matrix in matrix multiplication. The data type can be `torch.float8_e5m2`, `torch.float8_e4m3fn`, or `torch_npu.hifloat8`. This parameter must be a 2D tensor with shape `[K, M]`. The data layout can be `ND`.
- **`x2`** (`Tensor`): Required. Right matrix in matrix multiplication. The data type can be `torch.float8_e5m2`, `torch.float8_e4m3fn`, or `torch_npu.hifloat8`. This parameter must be a 2D tensor with shape `[K, N]`. The data layout can be `ND`.
- **`x2_scale`** (`Tensor`): Required. Scaling factor of the right matrix in the matrix multiplication. The data type can be `torch.float8_e8m0fnu` or `torch.float32`, and the shape must be 3D. When `torch.float8_e8m0fnu` is used, the optional parameter `x2_scale_dtype` must be set to the corresponding type. In this case, the dtype of `x2_scale` itself no longer takes effect, but the dtype of `x2_scale` itself must still be an 8-bit data type so that the shape is correct. The data layout can be ND.
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`x1_scale`** (`Tensor`): Optional. Scaling factor of the left matrix in the matrix multiplication. The data type can be `torch.float8_e8m0fnu` or `torch.float32`, and the shape must be 3D. When `torch.float8_e8m0fnu` is used, the optional parameter `x1_scale_dtype` must be set to the corresponding type. In this case, the dtype of `x1_scale` itself no longer takes effect, but the dtype of `x1_scale` itself must still be an 8-bit data type so that the shape is correct. The data layout can be ND.
- **`group_sizes`** (`List[int]`): Optional. The default value is `None`.
  - When the value is not `None`, only a 3D list is supported, in the form of [group\_m, group\_n, group\_k], which respectively indicate the quantization grouping on the m, n, and k dimensions. Taking group\_m as an example, group\_m values on the m dimension correspond to one quantization parameter.
  - When one or more of [group\_m, group\_n, group\_k] are 0, the API resets the value according to the input shapes of `x1`, `x2`, `x1_scale`, and `x2_scale`. Computation principle: If group\_m=0, the quantization grouping value in the m direction is inferred by the API. The inference formula is group\_m=m/scale\_m (m must be divisible by scale\_m), where m is the same as m in the x1 shape and scale\_m is the same as m in the x1\_scale shape.
  - Currently, the only value supported for [group\_m, group\_n, group\_k] in the mx quantization scenario is [1,1,32]. The T-T quantization scenario supports only [0,0,0].

- **`x1_scale_dtype`** (`int`): Optional. Used to explicitly specify the logical data type of `x1_scale`. None: default value, indicating that the actual input data type is the same as the dtype of the input `x1_scale`. Currently, only `torch.float8_e8m0fnu` is supported.

- **`x2_scale_dtype`** (`int`): Optional. Used to explicitly specify the logical data type of `x2_scale`. None: default value, indicating that the actual input data type is the same as the dtype of the input `x2_scale`. Currently, only `torch.float8_e8m0fnu` is supported.

## Return Values

**`self`** (`Tensor`): Final result matrix obtained by adding the result of the QuantBatchMatmul computation to the matrix to be accumulated. The supported data type, shape, and data layout must be the same as those of the input self.

## Constraints

- This API can be used in training scenarios.
- This API supports single-operator mode and TorchAir graph mode.
- Data type constraints

    | Scenario | x1 | x2 | x2_scale | x1_scale | self |
    | --- | --- | --- | --- | --- | --- |
    | mx quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float32` |
    | T-T quantization | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | `torch.float32` |

- Shape constraints

    | Scenario | x1 | x2 | x2_scale | x1_scale | self |
    | --- | --- | --- | --- | --- | --- |
    | mx quantization | (K,M) | (K,N) | (ceil(K/64),N,2) | (ceil(K/64),M, 2) | (M,N) |
    | T-T quantization | (K,M) | (K,N) | (1) | (1) | (M,N) |

## Example

- Single-operator call
  - Single-operator call for mx quantization

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
  - Graph mode call for mx quantization

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
