# torch_npu.npu_dynamic_quant_asymmetric

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function

- Description:

    Performs dynamic asymmetric quantization on input tensors. The `pertoken`, `pertensor`, and Mixture of Experts (MoE) scenarios are supported.

- Formulas:
    
    In `pertoken` scenarios, `rowMax` and `rowMin` indicate the maximum and minimum values by row. Row corresponds to the data of the last dimension of `x`, that is, a token. `DST_MAX` and `DST_MIN` correspond to the maximum and minimum values of the quantized `dtype`, respectively. The formulas are as follows:

    $$
    \text{scale} = \frac{\text{rowMax}(\mathbf{x}) - \text{rowMin}(\mathbf{x})}{DST\_MAX - DST\_MIN}\\
    \text{offset} = DST\_MAX - \frac{\text{rowMax}(\mathbf{x})}{\text{scale}}\\
    y = \text{round}(\frac{\mathbf{x}}{\text{scale}} + \text{offset})
    $$

    - If `Smooth Quant` is used in a non-MoE scenario, the `smooth_scales` input is introduced. The shape of `smooth_scales` is the same as that of the last dimension of `x`. Before quantization, `x` is multiplied by `smooth_scales` and then quantized according to the preceding formula. In the MoE scenario, both `smooth_scales` and `group_index` are introduced. In this case, `smooth_scales` contains multiple groups of smooth vectors, which are applied to different rows of `x` based on the values in `group_index`. Specifically, if `x` contains m tokens and `smooth_scales` has n rows, `smooth_scales[0]` is applied to `x[0:group_index[0]]`, and `smooth_scales[i]` is applied to `x[group_index[i-1]:group_index[i]]`, where i = 1, 2, ..., n-1.

## Prototype

```python
torch_npu.npu_dynamic_quant_asymmetric(x, *, smooth_scales=None, group_index=None, dst_type=None, quant_mode="pertoken") -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Source data tensor to be quantized. The data type can be `float16` or `bfloat16`. The data layout can be ND. Non-contiguous tensors are supported. The dimension of input `x` must be greater than 1. During int4 quantization, the last dimension of `x` must be a multiple of 8.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`smooth_scales`** (`Tensor`): Optional. Scaling tensor for `x`. The data type can be `float16` or `bfloat16`. The data layout can be ND. Non-contiguous tensors are supported.
    - In non-MoE scenarios, this parameter must be 1D and match the last dimension of `x`.
    - In MoE scenarios, this parameter must be 2D with shape `[E, H]`, where `E` indicates the number of experts ranging from 1 to 1024 (matching the first dimension of `group_index`), and `H` indicates the last dimension of `x`.
    - In single-operator mode, the `dtype` of `smooth_scales` must match that of `x`. In graph mode, they can be different.
- **`group_index`** (`Tensor`): Optional. Group index tensor for `smooth_scales` (representing the row index of `x`), valid only in MoE scenarios. The data type can be `int32`. The data layout can be ND. Non-contiguous tensors are supported. The shape must be `[E,]`, where `E` ranges from 1 to 1024 and matches the first dimension of `smooth_scales`. The tensor values must be strictly incrementing within the range [1, S], and the last value must equal `S` (`S` indicates the number of rows in the input `x`, which is the product of the dimensions of `x` except the last dimension).
- **`dst_type`** (`ScalarType`): Optional. Data type of the quantization output. Processed as `int8` if `None` is provided.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `int8` or `quint4x2`.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `int8` or `quint4x2`.
- **`quant_mode`** (`str`): Optional. Quantization mode. Valid values: `"pertoken"` or `"pertensor"`. The default value is `"pertoken"`. If `group_index` is not `None`, only "pertoken" is supported.

## Return Values

- **`y`** (`Tensor`): Quantized output tensor whose data type is specified by `dst_type`. When `dst_type` is `quint4x2`, the data type of `y` is `int32`, the last dimension of its shape is the last dimension of `x` divided by 8, and its other dimensions must match those of `x`, where each `int32` element contains eight `int4` results. In other scenarios, the shape of `y` must match that of the input `x`, and its data type is specified by `dst_type`.
- **`scale`** (`Tensor`): Scaling factors calculated during the asymmetric dynamic quantization process. The data type can be `float32`. If `quant_mode` is `"pertoken"`, the shape is the shape of `x` with its last dimension removed. If `quant_mode` is `"pertensor"`, the shape is `(1,)`.
- **`offset`** (`Tensor`): Offset factor calculated during the asymmetric dynamic quantization process. The data type can be `float32`, and its shape must match that of `scale`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- When using the optional parameters `smooth_scales`, `group_index`, and `dst_type`, you must pass them as keyword arguments.

## Examples

- Single-operator call
    - Perform `int8` quantization with only one input `x`.

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x)
        print(y, scale, offset)
        ```

    - Perform `int4` quantization with only one input `x`.

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, dst_type=torch.quint4x2)
        print(y, scale, offset)
        ```

    - Perform `int8` quantization with the `smooth_scales` input in non-MoE scenarios (without using `group_index`).

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        smooth_scales = torch.rand((8,), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, smooth_scales=smooth_scales)
        print(y, scale, offset)
        ```

    - Perform `int8` quantization with the `smooth_scales` input in MoE scenarios (using `group_index`).

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        smooth_scales = torch.rand((2, 8), dtype=torch.half).npu()
        group_index = torch.Tensor([1, 3]).to(torch.int32).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, smooth_scales=smooth_scales, group_index=group_index)
        print(y, scale, offset)
        ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    torch_npu.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    device=torch.device(f'npu:0')
    
    torch_npu.npu.set_device(device)
    
    class DynamicQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, input_tensor, smooth_scales=None, group_index=None, dst_type=None):
            out, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(input_tensor, smooth_scales=smooth_scales, group_index=group_index, dst_type=dst_type)
            return out, scale, offset
    
    x = torch.randn((2, 4, 6),device='npu',dtype=torch.float16).npu()
    smooth_scales = torch.randn((6),device='npu',dtype=torch.float16).npu()
    dynamic_quant_model = DynamicQuantModel().npu()
    dynamic_quant_model = torch.compile(dynamic_quant_model, backend=npu_backend, dynamic=True)
    out, scale, offset = dynamic_quant_model(x, smooth_scales=smooth_scales)
    print(out)
    print(scale)
    print(offset)
    ```
