# torch_npu.contrib.module.LinearWeightQuant

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>   |    √     |
| <term>Atlas inference products</term>    |    √     |

## Function

Encapsulates the `torch_npu.npu_weight_quant_batchmatmul` API to perform quantization on weight inputs and outputs in matrix multiplication computations. This class supports `pertensor`, `perchannel`, and `pergroup` scenarios.

Currently, <term>Atlas inference products</term> support only `perchannel` quantization.

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
- **`antiquant_group_size`** (`int`): Optional. Controls the group size in `pergroup` scenarios. The default value is 0. The value must be divisible by 32 in the range of `[32, K-1]`. <term>Atlas inference products</term>: This parameter is not supported currently.
- **`inner_precise`** (`int`):Computation mode. The default value is `0`. Valid values are `0` (high-precision mode) or `1` (high-performance mode, which may affect precision). In `pergroup` scenarios where `weight` is of `int32` type and in `FRACTAL_NZ` layout, and `M` is not greater than 16, this parameter can be set to `1` to improve performance. High-performance mode is not recommended in other scenarios.

**Computation Input**

**`x`** (`Tensor`): Input tensor (`x`) for matrix multiplication. The data layout can be ND. This parameter must be 2D with shape `(M, K)`.

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products</term>: The data type can be `float16` or `bfloat16`.
- <term>Atlas inference products</term>: Only `float16` is supported.

## Variable Description

- **`weight`** (`Tensor`): Weight tensor used for matrix multiplication. The data layout can be ND or FRACTAL_NZ. Non-contiguous tensors are supported. This parameter must be 2D with shape `(N, K)`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products</term>: The data type can be `int8` or `int32` (`int32` carries packed `int4` inputs. For details, see the call example of [torch_npu.npu_convert_weight_to_int4pack](../torch_npu/torch_npu-npu_convert_weight_to_int4pack.md)).
    - <term>Atlas inference products</term>: The data type can be `int8`. The FRACTAL_NZ layout is valid only in graph mode. Use `torchair.experimental.inference.use_internal_format_weight` to convert the data layout from ND to FRACTAL_NZ. For details, see [Examples](#section00001).

- **`antiquant_scale`** (`Tensor`): Scale used for weight dequantization. The data layout can be ND. Non-contiguous tensors are supported. This parameter must be 2D with shape `(N, 1)` or 1D with shape `(N,)` or `(1,)`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products</term>: The data type can be `float16`, `bfloat16`, or `int64`. In `pergroup` scenarios, the shape must be `(N, ceil_div(K, antiquant_group_size))`.
         - If the data type is `float16` or `bfloat16`, it must match that of `x`.
         - If the data type is `int64`, `x` must be of type `float16` and must not be transposed, and `weight` must be of type `int8` in ND layout and must be transposed. For details, see [Examples](#section00001). In this case, only `perchannel` scenarios are supported. The value range of `M` is `[1, 96]`, and both `K` and `N` must be aligned to 64.

    - <term>Atlas inference products</term>: The data type can be `float16`, and it must match that of `x`.

- **`antiquant_offset`** (`Tensor`): Offset used for weight dequantization. The data layout can be ND. Non-contiguous tensors are supported. This parameter must be 2D with shape `(N, 1)` or 1D with shape `(N,)` or `(1,)`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products</term>: The data type can be `float16`, `bfloat16`, or `int32`. In `pergroup` scenarios, the shape must be `(N, ceil_div(K, antiquant_group_size))`.
        - If the data type is `float16` or `bfloat16`, it must match that of `antiquant_scale`.
        - If the data type is `int32`, the data type of `antiquant_scale` must be `int64`.

    - <term>Atlas inference products</term>: Only `float16` is supported, and it must match that of `antiquant_scale`.

- **`quant_scale`** (`Tensor`): Scale used for output quantization. This parameter is supported only when `weight` layout is ND. The data layout can be ND. The data type can be `float32` or `int64`. This parameter must be 2D with shape `(1, N)` or 1D with shape `(N,)` or `(1,)`. When the data type of `antiquant_scale` is `int64`, this parameter must be omitted. <term>Atlas inference products</term>: This parameter is not supported currently.

- **`quant_offset`** (`Tensor`): Offset used for output quantization. This parameter is supported only when `weight` layout is ND. The data layout can be ND. The data type can be `float32`. This parameter must be 2D with shape `(1, N)` or 1D with shape `(N,)` or `(1,)`. When the data type of `antiquant_scale` is `int64`, this parameter must be omitted. <term>Atlas inference products</term>: This parameter is not supported currently.

- **`bias`** (`Tensor`): Bias tensor for matrix multiplication. The data layout can be ND. The data type can be `float16` or `float32`. Non-contiguous tensors are supported. This parameter must be 2D with shape `(1, N)` or 1D with shape `(N,)` or `(1,)`.
- **`antiquant_group_size`** (`int`): Group size for `pergroup` scenarios. The default value is `0`. The value must be divisible by 32 in the range of `[32, K-1]`. <term>Atlas inference products</term>: This parameter is not supported currently.

## Return Values

`Tensor`

Computation result. When `quant_scale` is provided, the data type of the output must be `int8`. If `quant_scale` is not provided, the data type of the output is identical to that of the input `x`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode. When the data layout of the input `weight` is FRACTAL_NZ, single-operator calls are currently not supported. Only graph mode calls are supported.
- The last two dimensions of `x` and `weight` must be `(M, K)` and `(N, K)`. The value range of `K` and `N` is `[1, 65535]`. When `x` is not transposed, the value range of `M` is [1, 2^31-1]. When `x` is transposed, the value range of M is [1, 65535].
- Empty tensor inputs are not supported.
- The input shapes of `antiquant_scale` and `antiquant_offset` must be identical.
- The input shapes of `quant_scale` and `quant_offset` must be identical, and `quant_offset` cannot exist independently of `quant_scale`.
- When the input type of `x` is `bfloat16`, the input type of `bias` must be `float32`. When the input type of `x` is `float16`, the input type of `bias` must be `float16`.
- To pass a `quant_scale` with a data type of `int64`, you must call the `torch_npu.npu_trans_quant_param` API in advance to convert the `quant_scale` and `quant_offset` from a data type of `float32` into a `quant_scale` input with a data type of `int64`. For details, see [Examples](#section00001).
- When the input `weight` has a data layout of FRACTAL_NZ and the data type is `int32`, `weight` must be transposed in `perchannel` scenarios. In `pergroup` scenarios, `x` must be transposed, `weight` must not be transposed, `antiquant_group_size` must be `64` or `128`, `K` must be aligned to `antiquant_group_size`, and `N` must be 64-aligned.

## Examples<a name="section00001"></a>

- Single-operator call

    ```python
    import torch
    import torch_npu
    from torch_npu.contrib.module import LinearWeightQuant
    x = torch.randn((8192, 320),device='npu',dtype=torch.float16)
    weight = torch.randn((320, 256),device='npu',dtype=torch.int8)
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
    weight = torch.randn((320, 256),device='npu',dtype=torch.int8)
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
    tng.experimental.inference.use_internal_format_weight(model) #: Converts weight from ND to FRACTAL_NZ
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    out = model(x)
    ```
