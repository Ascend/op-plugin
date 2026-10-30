# torch\_npu.npu\_grouped\_dynamic\_block\_quant

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- Function: quantizes each group at the granularity of basic blocks according to the start value of the passed group index (`group_list`), quantizing to (FP8/HiF8) and outputting the quantization parameter scale (`torch.float32`).
- Formulas:

$$
input\_max=block\_reduce\_max(abs(input)) \\
scale=min(input\_max/FP8\_MAX, 1/min\_scale) \\
y=cast\_to\_[HiF8/FP8](input/scale)
$$

## Prototype

```python
torch_npu.npu_grouped_dynamic_block_quant(input, group_list, *, min_scale=0.0, round_mode="rint", dst_type=291, row_block_size=1, 
                                          col_block_size=128, group_list_type=0, dst_type_max=0.0) -> (Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Input tensor, $input$ in the formula. The tensor must be 2D or 3D, with shape `[M, N]` or `[B, M, N]`. The data layout can be `ND`. The data type can be `torch.bfloat16` or `torch.float16`. Non-contiguous tensors are supported. Empty tensors are supported.
- **`group_list`** (`Tensor`): Required. Start index of the quantization group. The value must be greater than or equal to 0 and must be non-decreasing, and the last value must be equal to the size of the -2 axis of `input`. Only 1D is supported. The data layout can be ND. The data type can be `torch.int32`. Non-contiguous tensors are supported. Empty tensors are supported.
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`min_scale`** (`float`): Optional. Minimum value used in the scale computation, $min\_scale$ in the formulas. The value must be greater than or equal to 0. The default value is 0.0. The data type is `float32`.
- **`round_mode`** (`str`): Optional. Approximation mode used when casting the high-bit data to the target data type at the end. The default is "rint".
  - When dst_type is `torch.float8_e5m2`/`torch.float8_e4m3fn`, the supported mode is "rint".
  - When dst_type is `torch_npu.hifloat8`, the supported modes are "round" and "hybrid".

- **`dst_type`** (`int`): Optional. Data type of y after data conversion. The supported values are 290 (`torch_npu.hifloat8`), 291 (`torch.float8_e5m2`), and 292 (`torch.float8_e4m3fn`)/36. The default type is `torch.float8_e5m2`.
- **`row_block_size`** (`int`): Optional. Quantization granularity on the M axis. The supported values are 1, 128, 256, and 512. The default value is 1.
- **`col_block_size`** (`int`): Optional. Quantization granularity on the N axis. The supported values are 64, 128, 192, and 256. The default value is 128.
- **`group_list_type`** (`int`): Optional. Function type of group_list. The default value is 0, indicating that group_list is in cumsum mode.
- **`dst_type_max`** (`float`): Optional. Maximum positive value when the target quantization type is `torch_npu.hifloat8`. The value range is [0.0, 32768.0]. The default value is 0.0, indicating that the default maximum positive value of the `torch_npu.hifloat8` type is used.

## Return Values

- **`y`** (`Tensor`): Quantized output tensor, $y$ in the formula. The number of dimensions is the same as that of `input`. The data type can be `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`. Non-contiguous tensors are supported. Empty tensors are supported.
- **`scale`** (`Tensor`): Quantization scale for each group, $scale$ in the formula. The data type can be `torch.float32`. Non-contiguous tensors are supported. Empty tensors are supported. If `input` has shape `[M, N]` and `group_list` has shape `[g]`, `scale` has shape `[(M // row_block_size + g), (N / col_block_size)]`. If `input` has shape `[B, M, N]` and `group_list` has shape `[g]`, `scale` has shape `[B, (M // row_block_size + g), (N / col_block_size)]`.

## Constraints

- This API can be used in training and inference scenarios.
- This API supports single-operator mode and graph mode calls.

## Example

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
