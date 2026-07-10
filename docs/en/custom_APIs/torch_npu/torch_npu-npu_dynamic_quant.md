# torch_npu.npu_dynamic_quant

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function

- Description: Performs pertoken symmetric dynamic quantization on the input tensor.

    In Mixture of Experts (MoE) scenarios, `group_index` is introduced, and `smooth_scales` contains multiple groups of smoothing vectors that are applied to different rows of `x` according to the values specified in `group_index`. Specifically, if `x` contains m tokens and `smooth_scales` has n rows, `smooth_scales[0]` is applied to `x[0:group_index[0]]`, and `smooth_scales[i]` is applied to `x[group_index[i-1]:group_index[i]]`, where `i = 1, 2, ..., n-1`.

- Formulas:
    - When `smooth_scales` is omitted:
    $$
    \text{scale} = \frac{\text{rowMax}(\text{abs}(\mathbf{x}))}{DTYPE\_MAX} \\y = \text{round}\left(\frac{\mathbf{x}}{\text{scale}}\right)
    $$

    - When `smooth_scales` is provided:
    $$
    \text{scale} = \frac{\text{rowMax}(\text{abs}(\mathbf{x} * smooth\_scales))}{DTYPE\_MAX}  \\
    y = \text{round}\left(\frac{\mathbf{x} * smooth\_scales}{\text{scale}}\right)
    $$

    `rowMax` represents the maximum value calculated per row, and `DTYPE_MAX` is a constant representing the maximum value of the $y$ output data type.

## Prototype

```python
torch_npu.npu_dynamic_quant(x, *, smooth_scales=None, group_index=None, dst_type=None) ->(Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Source data tensor to be quantized. The data type can be `float16` or `bfloat16`. The data layout can be ND. Non-contiguous tensors are supported. The dimension of input `x` must be greater than 1. During int4 quantization, the last dimension of `x` must be a multiple of 8.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`smooth_scales`** (`Tensor`): Optional. Scaling tensor for `x`. The data type can be `float16` or `bfloat16`. The data layout can be ND. Non-contiguous tensors are supported. For shape constraints, see the Constraints section.
- **`group_index`** (`Tensor`): Optional. Group index tensor for `smooth_scales`, valid only in MoE scenarios. The data type can be `int32`. The data layout can be ND. Non-contiguous tensors are supported.

- **`dst_type`** (`ScalarType`): Optional. Data type of the quantization output. Processed as `int8` if `None` is provided.
    - Atlas A2 training products/Atlas A2 inference products: Valid values are `int8` or `quint4x2`.
    - Atlas A3 training products/Atlas A3 inference products: Valid values are `int8` or `quint4x2`.

## Return Values

- **`y`** (`Tensor`): Quantized output tensor whose data type is specified by `dst_type`. When `dst_type` is `quint4x2`, the data type of `y` is `int32`, the last dimension of its shape is the last dimension of `x` divided by 8, and its other dimensions must match those of `x`, where each `int32` element contains eight `int4` results. In other scenarios, the shape of `y` must match that of the input `x`, and its data type is specified by `dst_type`.
- **`scale`** (`Tensor`): Scaling factor computed during the symmetric dynamic quantization process. The data type is `float32`. The shape of this parameter matches that of `x` with its last dimension removed.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- This API supports MoE scenarios.

- When `smooth_scales` is provided:
    - When `group_index` is omitted: `smooth_scales` must be a 1D tensor, and its element count must be identical to the last dimension size of `x`.
    - When `group_index` is provided: `smooth_scales` must be a 2D tensor whose second dimension size is identical to the last dimension size of `x`. The `group_index` parameter must be a 1D tensor whose element count matches the first dimension size of `smooth_scales`. Elements in `group_index` must be monotonically increasing, and its final element value must equal the total element count of `x` divided by the size of its last dimension.

## Examples

- Single-operator call
    - When only one input `x` is provided:

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((3, 3), dtype=torch.float16).to("npu")
        >>> x
        tensor([[0.7261, 0.3726, 0.9126],
                [0.9023, 0.9990, 0.1279],
                [0.8628, 0.6240, 0.9028]], device='npu:0', dtype=torch.float16)
        >>>
        >>> output, scale = torch_npu.npu_dynamic_quant(x)
        >>> output
        tensor([[101,  52, 127],
                [115, 127,  16],
                [121,  88, 127]], device='npu:0', dtype=torch.int8)
        >>> scale
        tensor([0.0072, 0.0079, 0.0071], device='npu:0')
        ```

    - When `smooth_scales` is provided:

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((3, 3), dtype=torch.float16).to("npu")
        >>> x
        tensor([[0.6680, 0.9492, 0.0845],
                [0.1924, 0.5278, 0.1484],
                [0.6631, 0.9497, 0.0957]], device='npu:0', dtype=torch.float16)
        >>>
        >>> smooth_scales = torch.rand((3,), dtype=torch.float16).to("npu")
        >>> smooth_scales
        tensor([0.8042, 0.0884, 0.8901], device='npu:0', dtype=torch.float16)
        >>>
        >>> output, scale = torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales)
        >>> output
        tensor([[127,  20,  18],
                [127,  38, 108],
                [127,  20,  20]], device='npu:0', dtype=torch.int8)
        >>> scale
        tensor([0.0042, 0.0012, 0.0042], device='npu:0')
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

    device = torch.device(f'npu:0')
    torch_npu.npu.set_device(device)
    
    class DynamicQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, input_tensor, smooth_scales=None, group_index=None, dst_type=None):
            out, scale = torch_npu.npu_dynamic_quant(input_tensor, smooth_scales=smooth_scales, group_index=group_index,dst_type=dst_type)
            return out, scale

    x = torch.randn((2, 4, 6), device='npu', dtype=torch.float16).npu()
    smooth_scales = torch.randn((6), device='npu', dtype=torch.float16).npu()
    dynamic_quant_model = DynamicQuantModel().npu()
    dynamic_quant_model = torch.compile(dynamic_quant_model, backend=npu_backend, dynamic=True)
    out, scale = dynamic_quant_model(x, smooth_scales=smooth_scales)
    print(out)
    print(scale)
    
    # Expected output of the preceding code sample:
    tensor([[[-116,  127,   14, -105,   12,  -44],
            [   7, -127,  -49,  -27,   -4,   -7],
            [ -49,   18,  127,   39,   14,   13],
            [  12,  -47,  127,   73,   28,    1]],
    
            [[  62,  127,  -61,  -15,   -9,   -8],
            [ 127,  -74,  -66,  117,   27,   27],
            [   3,   65,   29,  127,  -27,   20],
            [  -4, -127,   13,  -40,  -21,  -11]]], device='npu:0',
        dtype=torch.int8)
    [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now,
    dtype cast replace with float. (function operator())
    tensor([[0.0080, 0.0422, 0.0219, 0.0132],
            [0.0176, 0.0069, 0.0093, 0.0368]], device='npu:0')
    ```
