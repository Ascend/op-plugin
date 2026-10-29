# torch_npu.npu_dynamic_quant_asymmetric

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

- Function:

    Performs dynamic asymmetric quantization on the input tensor. The pertoken, pertensor, perchannel, and MoE (mixture of experts) scenarios are supported.

- Formulas:
    
    rowMax and rowMin indicate the modes of obtaining the maximum and minimum values: In the `pertoken` mode, the maximum and minimum values are obtained by row (a row here corresponds to the data of the last dimension of `input`, that is, one token); in the `pertensor` mode, the maximum and minimum values of the entire tensor are obtained; in the `perchannel` mode, the maximum and minimum values are obtained by column. DST_MAX and DST_MIN respectively correspond to the maximum and minimum values of the dtype after quantization. The formulas are as follows:

    $$
    \text{scale} = \frac{\text{rowMax}(\mathbf{x}) - \text{rowMin}(\mathbf{x})}{DST\_MAX - DST\_MIN}\\
    \text{offset} = DST\_MAX - \frac{\text{rowMax}(\mathbf{x})}{\text{scale}}\\
    y = \text{round}(\frac{\mathbf{x}}{\text{scale}} + \text{offset})
    $$

    - In the MoE and non-MoE scenarios, if smooth quant is used, the smooth_scales input is introduced in the non-MoE scenario, and its shape is the same as the size of the last dimension of x. Before quantization, x is multiplied by smooth_scales and then quantized according to the preceding formulas. In the MoE scenario, both smooth_scales and group_index are introduced. In this case, smooth_scales contains multiple groups of smooth vectors, which are applied to different rows of x according to the values in group_index. Specifically, if x contains m tokens and smooth_scales has n rows, smooth_scales[0] is applied to x[0:group_index[0]], and smooth_scales[i] is applied to x[group_index[i-1]: group_index[i]], where i=1,2, ...,n-1.

## Prototype

```python
torch_npu.npu_dynamic_quant_asymmetric(input, *, smooth_scales=None, group_index=None, dst_type=None, quant_mode="pertoken", dst_type_max=0.0) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Source data tensor to be quantized. The data type can be `torch.float16` or `torch.bfloat16`, the data layout can be $ND$, and non-contiguous tensors are supported. The number of dimensions of the input `input` must be greater than 1. When `torch_npu.int4` quantization is performed, the last dimension of the `input` shape must be a multiple of 8.
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`smooth_scales`** (`Tensor`): Optional. A tensor that provides scaling factors (`scales`). The data type can be `torch.float16` or `torch.bfloat16`. The data layout can be ND. Non-contiguous tensors are supported.
    - In the non-MoE scenario, the shape must be 1D and equal to the last dimension of `input`.
    - In the MoE scenario, the shape is 2D [E, H]. E is the number of experts, and its value range is [1, 1024] and is the same as the first dimension of group_index. H is the last dimension of x.
    - In single-operator mode, the dtype of `smooth_scales` must be the same as that of `input`; in graph mode, they can be different.
- **`group_index`** (`Tensor`): Optional. Group index tensor for `smooth_scales` (representing the row index of `x`), valid only in MoE scenarios. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous tensors are supported. The shape must be `[E,]`, where `E` ranges from 1 to 1024 and matches the first dimension of `smooth_scales`. The tensor values must be strictly incrementing within the range [1, S], and the last value must equal `S` (`S` indicates the number of rows in the input `x`, which is the product of the dimensions of `x` except the last dimension).
- **`dst_type`** (`ScalarType`): Optional. Data type of the quantization output. Processed as `torch.int8` if `None` is provided.

    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data type can be `torch.int8` or `torch.quint4x2`.
    <!-- end id4 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950PR&950DT products</term>: the data type can be `torch.int8`, `torch.quint4x2`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`.
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
- **`scale`** (`Tensor`): Scaling factors calculated during the asymmetric dynamic quantization process. The data type is `torch.float32`. When `quant_mode` is `"pertoken"`, the shape is the shape of `input` with the last dimension removed. When `quant_mode` is `"perchannel"`, the shape is the shape of `input` with the second-to-last dimension removed and the last dimension kept identical to that of `input`. When `quant_mode` is `"pertensor"`, the shape is `(1,)`.
- **`offset`** (`Tensor`): Offset factor calculated during the asymmetric dynamic quantization process. The data type can be `torch.float32`, and its shape must match that of `scale`.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator mode and TorchAir graph mode.
- When the optional parameters `smooth_scales`, `group_index`, and `dst_type` are used, they must be passed by keyword.

## Example

- Single-operator call
    - Only one input `input` is provided for `torch.int8` quantization

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x)
        print(y, scale, offset)
        ```

    - Only one input `input` is provided for `torch_npu.int4` quantization

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, dst_type=torch.quint4x2)
        print(y, scale, offset)
        ```

    - The `smooth_scales` input is used in the non-MoE scenario (without `group_index`) for `torch.int8` quantization

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        smooth_scales = torch.rand((8,), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, smooth_scales=smooth_scales)
        print(y, scale, offset)
        ```

    - The `smooth_scales` input is used in the MoE scenario (with `group_index`) for `torch.int8` quantization

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
