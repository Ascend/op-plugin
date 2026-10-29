# torch_npu.contrib.module.LinearWeightQuant

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->

## Function

LinearWeightQuant is a wrapper class for the torch_npu.npu_weight_quant_batchmatmul API. It performs quantization on the weight input and the output in matrix multiplication and supports multiple quantization scenarios: pertensor, perchannel, and pergroup.

<!-- npu="310p" id15 -->
Currently, only perchannel quantization is supported for <term>Atlas inference products</term>.
<!-- end id15 -->

## Prototype

```python
torch_npu.contrib.module.LinearWeightQuant(in_features, out_features, bias=True, device=None, dtype=None, antiquant_offset=False, quant_scale=False, quant_offset=False, antiquant_group_size=0, inner_precise=0)
```

## Parameters

**Computation Parameters**

- **`in_features`** (`int`): Value of the K axis in fake-quantization matrix multiplication computation.
- **`out_features`** (`int`): Value of the N axis in fake-quantization matrix multiplication computation.
- **`bias`** (`bool`): Optional. Specifies whether to include bias in the computation. The default value is `True`. If set to `False`, bias is not added to the fake-quantization matrix multiplication computation.
- **`device`** (`string`): Optional. Device name for model execution. The default value is `None`.
- **`dtype`** (`torch.dtype`): Data type of the input `x` in the fake-quantization matrix multiplication computation. The default value is `None`.
- **`antiquant_offset`** (`bool`): Optional. Specifies whether to include the `antiquant_offset` parameter in the computation. The default value is `False`. If set to `False`, no offset is required during dequantization of the weight matrix.
- **`quant_scale`** (`bool`): Optional. Specifies whether to include the `quant_scale` parameter in the computation. The default value is `False`. If set to `False`, the fake-quantization output is not quantized.
- **`quant_offset`** (`bool`): Optional. Specifies whether to include the `quant_offset` parameter in the computation. The default value is `False`. If set to `False`, no offset is required during quantization of the fake-quantization output.
- **`antiquant_group_size`** (`int`): Optional. Controls the group size in `pergroup` scenarios. The default value is 0. The value must be divisible by 32 in the range of `[32, K-1]`. Atlas inference products: This parameter is not supported currently.
    <!-- npu="310p" id4 -->
    - <term>Atlas inference products</term>: This parameter is not supported currently.
    <!-- end id4 -->
- **`inner_precise`** (`int`): Computation mode. The default value is `0`. Valid values are `0` (high-precision mode) or `1` (high-performance mode, which may affect precision). In `pergroup` scenarios where `weight` is of `torch.int32` type and in `FRACTAL_NZ` layout, and `M` is not greater than 16, this parameter can be set to `1` to improve performance. High-performance mode is not recommended in other scenarios.

**Computation Inputs**

**x** (`Tensor`): Tensor type, that is, x in the matrix multiplication. The data layout can be $ND$. The input dimension can be 2D (M, K).

<!-- npu="A3,910b" id5 -->
- <term>Atlas A2 products</term> and <term>Atlas A3 training products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas inference products</term>: Only the `torch.float16` data type is supported.
<!-- end id6 -->

## Variable Description

- **`weight`** (`Tensor`): Weight tensor used for matrix multiplication. The data layout can be ND or FRACTAL_NZ. Non-contiguous tensors are supported. This parameter must be 2D with shape `(N, K)`.

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 training products</term>: The data type can be `torch.int8` or `torch.int32` (`torch.int32` carries packed `torch_npu.int4` inputs. For details, see the call example of [torch_npu.npu_convert_weight_to_int4pack](../torch_npu/torch_npu-npu_convert_weight_to_int4pack.md)).
    <!-- end id7 -->
    <!-- npu="310p" id8 -->
    - <term>Atlas inference products</term>: The data type can be `torch.int8`. The FRACTAL_NZ layout is valid only in graph mode. Use `torchair.experimental.inference.use_internal_format_weight` to convert the data layout from ND to FRACTAL_NZ. For details, see [Examples](#section00001).
    <!-- end id8 -->

- **`antiquant_scale`** (`Tensor`): Scale used for weight dequantization. The data layout can be ND. Non-contiguous tensors are supported. This parameter must be 2D with shape `(N, 1)` or 1D with shape `(N,)` or `(1,)`.

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 training products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int64`. In `pergroup` scenarios, the shape must be `(N, ceil_div(K, antiquant_group_size))`.
         - If the data type is `torch.float16` or `torch.bfloat16`, the data type must be the same as that of `x`.
         - If the data type is `torch.int64`, the data type of `x` must be `torch.float16` and the input must not be transposed. In addition, the data type of `weight` must be `torch.int8`, the data layout must be $ND$, and the input must be transposed. For details, see [Example](#section00001). In this case, only the perchannel scenario is supported, the value range of M is [1, 96], and K and N must be 64-aligned.
    <!-- end id9 -->
    <!-- npu="310p" id10 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`, and it must match that of `x`.
    <!-- end id10 -->

- **`antiquant_offset`** (`Tensor`): Offset used for weight dequantization. The data layout can be ND. Non-contiguous tensors are supported. This parameter must be 2D with shape `(N, 1)` or 1D with shape `(N,)` or `(1,)`.

    <!-- npu="A3,910b" id11 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 training products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int32`. In `pergroup` scenarios, the shape must be `(N, ceil_div(K, antiquant_group_size))`.
        - If the data type is `torch.float16` or `torch.bfloat16`, the data type must be the same as that of `antiquant_scale`.
        - If the data type is `torch.int32`, the data type of `antiquant_scale` must be `torch.int64`.
    <!-- end id11 -->
    <!-- npu="310p" id12 -->
    - <term>Atlas inference products</term>: Only the `torch.float16` data type is supported, and it must match that of `antiquant_scale`.
    <!-- end id12 -->

- **`quant_scale`** (`Tensor`): Scale used for output quantization. This parameter is supported only when `weight` layout is ND. The data layout can be ND. The data type can be `torch.float32` or `torch.int64`. This parameter must be 2D with shape `(1, N)` or 1D with shape `(N,)` or `(1,)`. When the data type of `antiquant_scale` is `torch.int64`, this parameter must be omitted. Atlas inference products: This parameter is not supported currently.
    <!-- npu="310p" id13 -->
    - <term>Atlas inference products</term>: This parameter is not supported currently.
    <!-- end id13 -->
- **`quant_offset`** (`Tensor`): Offset used for output quantization. This parameter is supported only when `weight` layout is ND. The data layout can be ND. The data type can be `torch.float32`. This parameter must be 2D with shape `(1, N)` or 1D with shape `(N,)` or `(1,)`. When the data type of `antiquant_scale` is `torch.int64`, this parameter must be omitted. Atlas inference products: This parameter is not supported currently.
    <!-- npu="310p" id14 -->
    - <term>Atlas inference products</term>: This parameter is not supported currently.
    <!-- end id14 -->
- **`bias`** (`Tensor`): Bias tensor for matrix multiplication. The data layout can be ND. The data type can be `torch.float16` or `torch.float32`. Non-contiguous tensors are supported. This parameter must be 2D with shape `(1, N)` or 1D with shape `(N,)` or `(1,)`.

## Return Values

`Tensor`

Represents the computation result. When `quant_scale` is present in the inputs, the output data type is `torch.int8`. When `quant_scale` is not present in the inputs, the output data type is the same as that of the input `x`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode. When the input `weight` is in FRACTAL_NZ format, single-operator calls are not supported currently and only graph mode calls are supported.
- The last two dimensions of `x` and `weight` must be in (M, K) and (N, K) format. The value range of K and N is [1, 65535]. When `x` is not transposed, the value range of M is [1, 2^31-1]. When `x` is transposed, the value range of M is [1, 65535].
- Empty tensor inputs are not supported.
- The input shapes of `antiquant_scale` and `antiquant_offset` must be the same.
- The input shapes of `quant_scale` and `quant_offset` must be the same, and `quant_offset` cannot exist independently of `quant_scale`.
- When the input type of `x` is `torch.bfloat16`, the input type of `bias` is `torch.float32`. When the input type of `x` is `torch.float16`, the input type of `bias` is `torch.float16`.
- To pass a `quant_scale` of the `torch.int64` data type, call the torch_npu.npu_trans_quant_param API in advance to convert the `quant_scale` and `quant_offset` of the `torch.float32` data type into a `quant_scale` input of the `torch.int64` data type. For details, see [Example](#section00001).
- When the input `weight` is in FRACTAL_NZ format and of the `torch.int32` type, the perchannel scenario requires `weight` to be a transposed input; the pergroup scenario requires `x` to be a transposed input, `weight` to be a non-transposed input, `antiquant_group_size` to be 64 or 128, K to be `antiquant_group_size`-aligned, and N to be 64-aligned.

## Example<a name="section00001"></a>

- Single-operator call

    ```python
    import torch
    import torch_npu
    from torch_npu.contrib.module import LinearWeightQuant
    x = torch.randn((8192, 320),device='npu',dtype=torch.float16)
    weight = torch.randint(-128, 128, (320, 256), device='npu', dtype=torch.int8)
    antiquantscale = torch.randn((1, 256),device='npu',dtype=torch.float16)
    antiquantoffset = torch.randn((1, 256),device='npu',dtype=torch.float16)
    quantscale = torch.randn((1, 256),device='npu',dtype=torch.float)
    quantoffset = torch.randn((1, 256),device='npu',dtype=torch.float)
    model = LinearWeightQuant(in_features=320,
                              out_features=256,
                              bias=False,
                              dtype=torch.float16,
                              antiquant_offset=True,
                              quant_scale=True,
                              quant_offset=True,
                              antiquant_group_size=0,
                              device=torch.device(f'npu:0')
                              )
    model.npu()
    model.weight.data = weight.transpose(-1, -2)
    model.antiquant_scale.data = antiquantscale.transpose(-1, -2)
    model.antiquant_offset.data = antiquantoffset.transpose(-1, -2)
    model.quant_scale.data = torch_npu.npu_trans_quant_param(quantscale, quantoffset)
    model.quant_offset.data = quantoffset
    out = model(x)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torch_npu.contrib.module import LinearWeightQuant
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    x = torch.randn((8192, 320),device='npu',dtype=torch.bfloat16)
    weight = torch.randint(-128, 128, (320, 256), device='npu', dtype=torch.int8)
    antiquantscale = torch.randn((1, 256),device='npu',dtype=torch.bfloat16)
    antiquantoffset = torch.randn((1, 256),device='npu',dtype=torch.bfloat16)
    quantscale = torch.randn((1, 256),device='npu',dtype=torch.float)
    quantoffset = torch.randn((1, 256),device='npu',dtype=torch.float)
    model = LinearWeightQuant(in_features=320,
                              out_features=256,
                              bias=False,
                              dtype=torch.bfloat16,
                              antiquant_offset=True,
                              quant_scale=True,
                              quant_offset=True,
                              antiquant_group_size=0,
                              device=torch.device(f'npu:0')
                              )
    model.npu()
    model.weight.data = weight.transpose(-1, -2)
    model.antiquant_scale.data = antiquantscale.transpose(-1, -2)
    model.antiquant_offset.data = antiquantoffset.transpose(-1, -2)
    model.quant_scale.data = quantscale
    model.quant_offset.data = quantoffset
    tng.experimental.inference.use_internal_format_weight(model) # Convert the ND weight input to FRACTAL_NZ format
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    out = model(x)
    ```
