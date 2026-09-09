# torch\_npu.npu\_grouped\_dynamic\_block\_quant

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## Function

- Quantizes each group at block granularity based on the starting values of the group indices (`group_list`) to FP8/HiFP8 and outputs the quantization parameter `scale` (FP32).

- Formulas:

$$
input\_max=block\_reduce\_max(abs(input)) \\
scale=min(input\_max/FP8\_MAX, 1/min\_scale) \\
y=cast\_to\_[HiF8/FP8](input/scale)
$$

## Prototype

```python
torch_npu.npu_grouped_dynamic_block_quant(input, group_list, *, min_scale=0.0, round_mode="rint", dst_type=291, row_block_size=1, col_block_size=128, group_list_type=0) -> (Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Input tensor, $input$ in the formula. The tensor must be 2D or 3D, with shape `[M, N]` or `[B, M, N]`. The data layout can be `ND`. The data type can be `bfloat16` or `float16`. Non-contiguous tensors are supported. Empty tensors are supported.
- **`group_list`** (`Tensor`): Required. Starting indices of the quantization groups. Values must be greater than or equal to `0` and non-decreasing, and the last value must equal the size of the second-to-last dimension of `input`. The tensor must be 1D. The data layout can be `ND`. The data type can be `int32`. Non-contiguous tensors are supported. Empty tensors are supported.
- **`*`**: Position delimiter. Variables before this delimiter are position-dependent and must be passed in order. Variables after this delimiter are optional keyword arguments and must be assigned using key-value pairs. If not specified, their default values are used.
- **`min_scale`** (`float`): Optional. Minimum value used in computing `scale`, $min\_scale$ in the formula. The value must be greater than or equal to `0`. The default value is `0.0`. The data type can be `float32`.
- **`round_mode`** (`str`): Optional. Approximation mode used when casting from the higher-bit data type to the target data type. The default value is `"rint"`.
  - When `dst_type` is `float8_e5m2` or `float8_e4m3fn`, `"rint"` is supported.
  - When `dst_type` is `hifloat8`, `"round"` and `"hybrid"` are supported.

- **`dst_type`** (`int`): Optional. Data type of `y` after data conversion. Supported values are `290` (`hifloat8`), `291` (`float8_e5m2`), and `292`/`36` (`float8_e4m3fn`). The default data type is `float8_e5m2`.
- **`row_block_size`** (`int`): Optional. Quantization granularity along the `M` axis. Currently supported values are `1`, `128`, `256`, and `512`. The default value is `1`.
- **`col_block_size`** (`int`): Optional. Quantization granularity along the `N` axis. Currently supported values are `64`, `128`, `192`, and `256`. The default value is `128`.
- **`group_list_type`** (`int`): Optional. Function type of `group_list`. The default value is `0`, indicating that `group_list` is in cumulative-sum mode.

## Return Values

- **`y`** (`Tensor`): Quantized output tensor, $y$ in the formula. The number of dimensions is the same as that of `input`. The data type can be `hifloat8`, `float8_e5m2`, or `float8_e4m3fn`. Non-contiguous tensors are supported. Empty tensors are supported.
- **`scale`** (`Tensor`): Quantization scale for each group, $scale$ in the formula. The data type can be `float32`. Non-contiguous tensors are supported. Empty tensors are supported. If `input` has shape `[M, N]` and `group_list` has shape `[g]`, `scale` has shape `[(M // row_block_size + g), (N / col_block_size)]`. If `input` has shape `[B, M, N]` and `group_list` has shape `[g]`, `scale` has shape `[B, (M // row_block_size + g), (N / col_block_size)]`.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator and graph mode calls.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import numpy as np
    
    def grouped_dynamic_block_quant_test(x_dtype, dst_type):
        # Construct the x tensor
        x = torch.randn((1, 128), dtype=x_dtype).npu()
        group_list = torch.ones((1,), dtype=torch.int32).npu()
        y_tmp, scale_tmp = torch_npu.npu_grouped_dynamic_block_quant(x, group_list, dst_type=dst_type)
        y = y_tmp.cpu()
        scale = scale_tmp.cpu()
        print("GroupedDynamicBlockQuant result:")
        print("x:\n", x)
        print("group_list:\n", group_list)
        print("y:\n", y)
        print("scale:\n", scale)
    
    if __name__ == "__main__":
        grouped_dynamic_block_quant_test(torch.float16, torch.float8_e5m2)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair
    import numpy as np
    
    class GroupedDynamicBlockQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, group_list, min_scale=0.0, dst_type=torch_npu.float8_e5m2, row_block_size=1, col_block_size=128, group_list_type=0):
            return torch_npu.npu_grouped_dynamic_block_quant(x, group_list, min_scale=min_scale, dst_type=dst_type, row_block_size=row_block_size, col_block_size=col_block_size, group_list_type=group_list_type)
    
    def grouped_dynamic_block_quant_test(x_dtype, dst_type):
        # Construct the x tensor
        x = torch.randn((1, 128), dtype=x_dtype).npu()
        group_list = torch.ones((1,), dtype=torch.int32).npu()
        model = GroupedDynamicBlockQuantModel()
        model.to('npu')
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        y_tmp, scale_tmp = model(x, group_list, dst_type=dst_type)
        y = y_tmp.cpu()
        scale = scale_tmp.cpu()
        print("GroupedDynamicBlockQuant result:")
        print("x:\n", x)
        print("y:\n", y)
        print("scale:\n", scale)
    
    if __name__ == "__main__":
        grouped_dynamic_block_quant_test(torch.float16, torch.float8_e5m2)
    ```
