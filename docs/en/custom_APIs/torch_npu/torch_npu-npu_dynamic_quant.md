# torch_npu.npu_dynamic_quant

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- **Function**: performs dynamic quantization on the input tensor. The pertoken, pertensor, perchannel, and MoE (mixture of experts) scenarios are supported.

    In the MoE (mixture of experts) scenario, `group_index` is introduced. `smooth_scales` contains multiple groups of smooth vectors, which are applied to different rows of `input` according to the values in `group_index`. Specifically, if `input` contains m tokens and `smooth_scales` has n rows, `smooth_scales[0]` is applied to `input[0:group_index[0]]`, and `smooth_scales[i]` is applied to `input[group_index[i-1]: group_index[i]]`, where `i=1, 2, ..., n-1`.

    The quantization mode can be specified through `quant_mode`. The `pertoken` (default), `perchannel`, and `pertensor` modes are supported. `pertoken` indicates quantization at token granularity, `perchannel` indicates quantization by channel, and `pertensor` indicates that the same scale is used to quantize the entire tensor.
    
    <!-- npu="950" id11 -->
    The `perchannel` and `pertensor` modes are supported only on <term>Ascend 950PR&950DT products</term>. For details, see the parameter description.

    <!-- end id11 -->
- Formulas:
    - If `smooth_scales` does not exist:
    $$
    \text{scale} = \frac{\text{rowMax}(\text{abs}(\mathbf{x}))}{DTYPE\_MAX} \\
    y = \text{round}\left(\frac{\mathbf{x}}{\text{scale}}\right)
    $$

    - If `smooth_scales` exists:
    $$
    \text{scale} = \frac{\text{rowMax}(\text{abs}(\mathbf{x} * smooth\_scales))}{DTYPE\_MAX}  \\
    y = \text{round}\left(\frac{\mathbf{x} * smooth\_scales}{\text{scale}}\right)
    $$

    rowMax indicates the mode of obtaining the maximum value: in the `pertoken` mode, it indicates obtaining the maximum value of a row; in the `pertensor` mode, it indicates obtaining the maximum value of the entire tensor; in the `perchannel` mode, it indicates obtaining the maximum value of a column. DTYPE_MAX indicates a constant, which is the maximum value of the data type corresponding to the y output. When `dst_type_max` is not 0, the value of `dst_type_max` is used as DTYPE_MAX.

## Prototype

```python
torch_npu.npu_dynamic_quant(input, *, smooth_scales=None, group_index=None, dst_type=None, quant_mode="pertoken", dst_type_max=0.0) ->(Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Source data tensor to be quantized. The data type can be `torch.float16` or `torch.bfloat16`, the data layout can be $ND$, and non-contiguous tensors are supported. The number of dimensions of the input `input` must be greater than 1. When `torch_npu.int4` quantization is performed, the last dimension of the `input` shape must be a multiple of 8.
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`smooth_scales`** (`Tensor`): Optional. Scaling tensor for `x`. The data type can be `torch.float16` or `torch.bfloat16`. The data layout can be ND. Non-contiguous tensors are supported. For shape constraints, see the Constraints section.
- **`group_index`** (`Tensor`): Optional. Group index tensor for `smooth_scales`, valid only in MoE scenarios. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous tensors are supported. `group_index` is a 1D tensor whose number of elements is the same as the first dimension of `smooth_scales`. When `group_index` is not `None`, `smooth_scales` must not be `None`.
- **`dst_type`** (`ScalarType`): Optional. Data type of the quantization output. Processed as `torch.int8` if `None` is provided.
    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the supported values are `torch.int8` and `torch.quint4x2`.
    <!-- end id4 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950PR&950DT products</term>: the supported values are `torch.int8`, `torch.quint4x2`, `torch_npu.hifloat8`, `torch.float8_e5m2`, and `torch.float8_e4m3fn`.
    <!-- end id6 -->

- **`quant_mode`** (`str`): Optional. Specifies the quantization mode. The default value is `"pertoken"`. If `group_index` is not `None`, only `"pertoken"` is supported.

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: this parameter is not supported currently, and `"pertoken"` is used by default.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>: the supported values are `"pertoken"` (token granularity), `"perchannel"` (channel granularity), and `"pertensor"` (one scale shared by the entire tensor).
    <!-- end id8 -->

- **`dst_type_max`** (`float`): Optional. Maximum representable value of the target data type. The default value is 0.0.

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: this parameter is not supported currently.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>: it takes effect only when `dst_type` is `torch_npu.hifloat8`, and the supported values are 0.0 to 32768.0. When the value is 0.0, the maximum value representable by the target precision is used; when the value is not 0.0, the passed value is used as the maximum value of the target data type.
    <!-- end id10 -->

## Return Values

- **`y`** (`Tensor`): Quantized output tensor whose data type is specified by `dst_type`. When `dst_type` is `torch.quint4x2`, the data type of `y` is `torch.int32`, the last dimension of its shape is the last dimension of `x` divided by 8, and its other dimensions must match those of `x`, where each `torch.int32` element contains eight `torch_npu.int4` results. In other scenarios, the shape of `y` must match that of the input `x`, and its data type is specified by `dst_type`.
    <!-- npu="950" id5 -->
    - On <term>Ascend 950PR&950DT products</term>, when `dst_type` is `torch_npu.hifloat8`, the data type of `y` is `torch.uint8` (which actually carries the `torch_npu.hifloat8` type).
    <!-- end id5 -->
- **`scale`** (`Tensor`): Scaling factor computed during the symmetric dynamic quantization process. The data type is `torch.float32`. The shape of this parameter matches that of `x` with its last dimension removed.
  - When `quant_mode` is `"pertoken"`, the shape is the shape of `input` with the last dimension removed.
  - When `quant_mode` is `"perchannel"`, the shape is the shape of `input` with the second-to-last dimension removed, and the last dimension remains the same as that of `input`.
  - When `quant_mode` is `"pertensor"`, the shape is `(1,)`.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator mode and TorchAir graph mode.
- This API supports the MoE scenario.

- When `smooth_scales` is used:
    - In the `pertoken`/`pertensor` mode without `group_index`, `smooth_scales` must be a 1D Tensor, and the number of elements must be the same as the size of the last dimension of `input`.
    - In the `perchannel` mode, `smooth_scales` must be a 1D Tensor, and the number of elements must be the same as the size of the second-to-last dimension of `input`.
    - If `group_index` is used, `smooth_scales` must be a 2D Tensor. The value range of the first dimension (number of experts) is `[1, 1024]`, and the number of elements of the second dimension must be the same as the size of the last dimension of `input`. `group_index` must be a 1D array, and its number of elements must be the same as the first dimension of `smooth_scales`. The elements in `group_index` must be monotonically increasing, and the value of its last element must be equal to the number of elements of `input` divided by the last dimension of `input`.
    - In single-operator mode, the data type of `smooth_scales` must be the same as that of `input`; in graph mode, they can be different.

## Example

- Single-operator call
    - Only one input `input`

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((3, 3), dtype=torch.float16).to("npu")
        >>> print(x)
        tensor([[0.7261, 0.3726, 0.9126],
                [0.9023, 0.9990, 0.1279],
                [0.8628, 0.6240, 0.9028]], device='npu:0', dtype=torch.float16)
        >>>
        >>> output, scale = torch_npu.npu_dynamic_quant(x)
        >>> print(output)
        tensor([[101,  52, 127],
                [115, 127,  16],
                [121,  88, 127]], device='npu:0', dtype=torch.int8)
        >>> print(scale)
        tensor([0.0072, 0.0079, 0.0071], device='npu:0')
        ```

    - Using the `smooth_scales` input

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.rand((3, 3), dtype=torch.float16).to("npu")
        >>> print(x)
        tensor([[0.6680, 0.9492, 0.0845],
                [0.1924, 0.5278, 0.1484],
                [0.6631, 0.9497, 0.0957]], device='npu:0', dtype=torch.float16)
        >>>
        >>> smooth_scales = torch.rand((3,), dtype=torch.float16).to("npu")
        >>> print(smooth_scales)
        tensor([0.8042, 0.0884, 0.8901], device='npu:0', dtype=torch.float16)
        >>>
        >>> output, scale = torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales)
        >>> print(output)
        tensor([[127,  20,  18],
                [127,  38, 108],
                [127,  20,  20]], device='npu:0', dtype=torch.int8)
        >>> print(scale)
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
