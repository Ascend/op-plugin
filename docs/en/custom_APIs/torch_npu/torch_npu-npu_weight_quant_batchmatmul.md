# torch\_npu.npu\_weight\_quant\_batchmatmul<a name="en-us_topic_0000002231202136"></a>

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>     |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |
|<term>Atlas inference accelerator cards</term> | √   |

## Function

- Description: Performs matrix multiplication with quantization support for the `weight` input and the output. `pertensor`, `perchannel`, and `pergroup` quantization modes are supported.
- Formulas:

     $$
     y = x @ ANTIQUANT(weight) + bias 
     $$
     In the formulas, $weight$ represents the input in the fake-quantization scenario, and its dequantization formula $\text{ANTIQUANT}(\text{weight})$ is:
     $$
     ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
     $$
     When `quant_scale` is provided, the output is quantized using the following formula:
     $$
     y = QUANT(x @ ANTIQUANT(weight) + bias) \\
     = (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset
     $$
     When `quant_scale` is `None`, the final computation result is directly output as:
     $$
     y = x @ ANTIQUANT(weight) + bias
     $$

## Prototype

```python
torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset=None, quant_scale=None, quant_offset=None, bias=None, antiquant_group_size=0, inner_precise=0) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. The left matrix in matrix multiplication. $x$ in the formulas. The data layout can be ND. Transposed non-contiguous tensors are supported. This parameter must be 2D with shape $(M, K)$.
    - <term>Atlas inference accelerator cards</term>: The data type can be `float16`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float16` or `bfloat16`.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be `float16` or `bfloat16`.

- **`weight`** (`Tensor`): Required. The right matrix in matrix multiplication. $weight$ in the formulas. Transposed non-contiguous tensors are supported. This parameter must be 2D with shape $(K, N)$. The dimensions must match those of `x`. When the data layout is ND, using a transposed `weight` input is recommended in `perchannel` scenarios to improve performance.
    - <term>Atlas inference accelerator cards</term>: The data type can be `int8`. The data layout can be ND or FRACTAL_NZ, where the FRACTAL_NZ layout is valid only in graph mode and requires the `torch_npu.npu_format_cast` API to convert data from ND to FRACTAL_NZ. For details, see [Examples](#en-us_topic_0000001771071862_section14459801435).
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `int8` or `int32` (where `int32` carries `int4` inputs; for details, see the call example of [torch_npu.npu_convert_weight_to_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)). The data layout can be ND or FRACTAL_NZ.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be `int8` or `int32` (where `int32` carries `int4` inputs; for details, see the call example of [torch_npu.npu_convert_weight_to_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)). The data layout can be ND or FRACTAL_NZ.

- **`antiquant_scale`** (`Tensor`): Required. Dequantization scale factor used for transposing and dequantizing the `weight` matrix. $antiquantScale$ in the dequantization formula. The data layout can be ND. Transposed non-contiguous tensors are supported. The supported shapes of `antiquant_scale` depend on the quantization mode:

    - `per_tensor` mode: The input shape is `(1,)` or `(1, 1)`.
    - `per_channel` mode: The input shape is `(1, N)` or `(N,)`.
    - `per_group` mode: The input shape is `(ceil(K, antiquant_group_size), N)`.

    The following data types are supported for `antiquant_scale`:

    - <term>Atlas inference accelerator cards</term>: The data type can be `float16`, which must match that of `x`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float16`, `bfloat16`, or `int64`.
        - If the input is `float16` or `bfloat16`, its data type must match that of `x`.
        - If the input is `int64`, the data type of `x` must be `float16` and `x` must not be transposed, and `weight` must be a transposed `int8` tensor with a data layout of ND. For details, see [Examples](#en-us_topic_0000001771071862_section14459801435). In this case, only the `perchannel` scenario is supported, the value range of `M` is [1, 96], and both `K` and `N` must be aligned to 64.

    - <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be `float16`, `bfloat16`, or `int64`.
        - If the input is `float16` or `bfloat16`, its data type must match that of `x`.
        - If the input is `int64`, the data type of `x` must be `float16` and `x` must not be transposed, and `weight` must be a transposed `int8` tensor with a data layout of ND. For details, see [Examples](#en-us_topic_0000001771071862_section14459801435). In this case, only the `perchannel` scenario is supported, the value range of `M` is [1, 96], and both `K` and `N` must be aligned to 64.

- **`antiquant_offset`** (`Tensor`): Optional. Dequantization offset used for dequantizing the `weight` matrix. $antiquantOffset$ in the dequantization formula. The default value is `None`. The data layout can be ND. Transposed non-contiguous tensors are supported. This parameter can be 1D with shape `(N,)` or `(1,)`, or 2D with shape `(1, N)`.
    - <term>Atlas inference accelerator cards</term>: The data type can be `float16`, which must match that of `antiquant_scale`. In `pergroup` scenarios, the shape must be `(ceil_div(K, antiquant_group_size), N)`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float16`, `bfloat16`, or `int32`. In `pergroup` scenarios, the shape must be `(ceil_div(K, antiquant_group_size), N)`.
        - If the input is `float16` or `bfloat16`, its data type must match that of `antiquant_scale`.
        - If the input is `int32`, the data type of `antiquant_scale` must be `int64`.

    - <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be `float16`, `bfloat16`, or `int32`. In `pergroup` scenarios, the shape must be `(ceil_div(K, antiquant_group_size), N)`.
        - If the input is `float16` or `bfloat16`, its data type must match that of `antiquant_scale`.
        - If the input is `int32`, the data type of `antiquant_scale` must be `int64`.

- **`quant_scale`** (`Tensor`): Optional. Quantization scale factor used for quantizing the output matrix. The default value is `None`. This parameter is supported only when the data layout of `weight` is ND. The data type can be `float32` or `int64`. The data layout can be ND. This parameter can be 1D with shape `(N,)` or `(1,)`, or 2D with shape `(1, N)`. When the data type of `antiquant_scale` is `int64`, this parameter must be omitted.
    - <term>Atlas inference accelerator cards</term>: This parameter is not supported currently.

- **`quant_offset`** (`Tensor`): Optional. Quantization offset used for quantizing the output matrix. $quantOffset$ in the quantization formula. The default value is `None`. This parameter is supported only when the data layout of `weight` is ND. The data type can be `float32`. The data layout can be ND. This parameter can be 1D with shape `(N,)` or `(1,)`, or 2D with shape `(1, N)`. When the data type of `antiquant_scale` is `int64`, this parameter must be omitted.
    - <term>Atlas inference accelerator cards</term>: This parameter is not supported currently.

- **`bias`** (`Tensor`): Optional. Bias term in matrix multiplication. $bias$ in the formulas. The default value is `None`. The data layout can be ND. Non-contiguous tensors are not supported. This parameter can be 1D with shape `(N,)` or 2D with shape `(1, N)`.
    - <term>Atlas inference accelerator cards</term>: The data type can be `float16`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float16` or `bfloat32`. When the data type of `x` is `bfloat16`, the data type of `bias` must be `float32`. When the data type of `x` is `float16`, the data type of `bias` must be `float16`.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be `float16` or `bfloat32`. When the data type of `x` is `bfloat16`, the data type of `bias` must be `float32`. When the data type of `x` is `float16`, the data type of `bias` must be `float16`.

- **`antiquant_group_size`** (`int`): Optional. Controls the group size in `pergroup` quantization scenarios. This parameter does not take effect in other quantization scenarios. The default value is `0`. In `pergroup` scenarios, the value must be a multiple of 32 within the range `[32, K - 1]`.
- **`inner_precise`** (`int`): Optional. Computation mode. The default value is `0`. Valid values are `0` (high-precision mode) or `1` (high-performance mode, which may affect precision). When `weight` is an `int32` tensor with a data layout of FRACTAL_NZ, this parameter can be set to `1` in `pergroup` scenarios (where $M \le 16$) to improve performance. High-performance mode is not recommended in other scenarios.

## Return Values<a name="en-us_topic_0000001771071862_section22231435517"></a>

`Tensor`

When `quant_scale` is provided, the data type of the output must be `int8`. If `quant_scale` is not provided, the data type of the output is identical to that of the input `x`.

## Constraints<a name="en-us_topic_0000001771071862_section12345537164214"></a>

- This API can be used in inference scenarios.
- This API supports graph mode. When the data layout of the input `weight` is FRACTAL_NZ, single-operator calls are currently not supported. Only graph mode calls are supported.
- The last two dimensions of `x` and `weight` must follow the `(M, K)` and `(K, N)` formats, respectively. The value range of `K` and `N` is `[1, 65535]`. When `x` is not transposed, the value range of `M` is [1, 2^{31} - 1]. When `x` is transposed, the value range of M is [1, 65535].
- Empty tensor inputs are not supported.
- The input shapes of `antiquant_scale` and `antiquant_offset` must be identical.
- The input shapes of `quant_scale` and `quant_offset` must be identical, and `quant_offset` cannot exist independently of `quant_scale`.
- To pass a `quant_scale` with a data type of `int64`, you must call the `torch_npu.npu_trans_quant_param` API in advance to convert the `quant_scale` and `quant_offset` from a data type of `float32` into a `quant_scale` input with a data type of `int64`. For details, see [Examples](#en-us_topic_0000001771071862_section14459801435).
- When the input `weight` has a data layout of FRACTAL_NZ and the data type is `int32`, `weight` must be transposed in `perchannel` scenarios; whereas in `pergroup` scenarios, `x` must be transposed, `weight` must not be transposed, `antiquant_group_size` must be `64` or `128`, `K` must be aligned to `antiquant_group_size`, and `N` must be aligned to 64.
- When the input `weight` shape is `(1, 8)` and the data type is `int4`, `weight` must not be transposed. Otherwise, an error will be raised indicating that the K axes of the `x` matrix and `weight` matrix do not match. In this scenario, you are advised to use non-quantization operators to achieve higher model accuracy and performance.
- When `antiquant_scale` is `float16` or `bfloat16`, the data types of `x` and `antiquant_scale` must be identical in single-operator mode and can differ in graph mode. If the data types differ, the API internally determines whether to convert them to a unified data type. You can dump graph information to inspect the actual data types involved in the computation.

## Examples<a name="en-us_topic_0000001771071862_section14459801435"></a>

- Single-operator call
    - When `weight` is not transposed and `quant_scale` is provided, only the following products are supported:

        - <term>Atlas A2 training products/Atlas A2 inference products</term>
        - <term>Atlas A3 training products/Atlas A3 inference products</term>

            ```python
            import torch
            import torch_npu
            # When the input data type is int8, and the layout is ND:
            cpu_x = torch.randn((8192, 320),dtype=torch.float16)
            cpu_weight = torch.randint(low=-8, high=8, size=(320, 256),dtype=torch.int8)
            cpu_antiquantscale = torch.randn((1, 256),dtype=torch.float16)
            cpu_antiquantoffset = torch.randn((1, 256),dtype=torch.float16)
            cpu_quantscale = torch.randn((1, 256),dtype=torch.float32)
            cpu_quantoffset = torch.randn((1, 256),dtype=torch.float32)
            quantscale= torch_npu.npu_trans_quant_param(cpu_quantscale.npu(), cpu_quantoffset.npu())
            npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(),quantscale.npu())
            ```

    - When `weight` is transposed and `antiquant_scale` is provided, only the following products are supported:

        - <term>Atlas A2 training products/Atlas A2 inference products</term>
        - <term>Atlas A3 training products/Atlas A3 inference products</term>
        - <term>Atlas inference accelerator cards</term>

            ```python
            import torch
            import torch_npu
            cpu_x = torch.randn((96, 320),dtype=torch.float16)
            cpu_weight = torch.randint(low=-8, high=8, size=(256, 320),dtype=torch.int8)
            cpu_antiquantscale = torch.randn((256),dtype=torch.float16)
            # Construct the `scale` parameter with an `int64` data type.
            antiquant_scale = torch_npu.npu_trans_quant_param(cpu_antiquantscale.to(torch.float32).npu()).reshape(256, 1)
            cpu_antiquantoffset = torch.randint(-128, 127, (256, 1), dtype=torch.int32)
            npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.transpose(-1,-2).npu(), antiquant_scale.transpose(-1,-2).npu(), cpu_antiquantoffset.transpose(-1,-2).npu())
            ```

    - When `weight` is transposed and `antiquant_scale` is provided, only the following products are supported:

        - <term>Atlas A2 training products/Atlas A2 inference products</term>
        - <term>Atlas A3 training products/Atlas A3 inference products</term>

            ```python
            import torch
            import torch_npu
            # When the input data type is int8, and the layout is ND:
            cpu_x = torch.randn((96, 320),dtype=torch.float16)
            cpu_weight = torch.randint(low=-8, high=8, size=(256, 320),dtype=torch.int8)
            cpu_antiquantscale = torch.randn((256,1),dtype=torch.float16)
            cpu_antiquantoffset = torch.randint(-128, 127, (256,1), dtype=torch.float16)
            npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.npu().transpose(-1, -2), cpu_antiquantscale.npu().transpose(-1, -2), cpu_antiquantoffset.npu().transpose(-1, -2))
            ```

- Graph mode call
    - When data layout of `weight` is ND:

        ```python
        # Graph mode
        import torch
        import torch_npu
        import  torchair as tng
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.debug.graph_dump.type = "pbtxt"
        npu_backend = tng.get_npu_backend(compiler_config=config)
        
        cpu_x = torch.randn((8192, 320),device='npu',dtype=torch.bfloat16)
        cpu_weight = torch.randint(low=-8, high=8, size=(320, 256), dtype=torch.int8, device='npu')
        cpu_antiquantscale = torch.randn((1, 256),device='npu',dtype=torch.bfloat16)
        cpu_antiquantoffset = torch.randn((1, 256),device='npu',dtype=torch.bfloat16)
        
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
        
            def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
                return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale ,quant_offset, bias, antiquant_group_size)
        
        cpu_model = MyModel()
        model = cpu_model.npu()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        npu_out = model(cpu_x.npu(), cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(), None, None, None, 0)
        ```

    - If the data layout of the input `weight` is FRACTAL_NZ, only <term>Atlas inference accelerator cards</term> are supported.

        ```python
        import torch_npu
        import torch
        from torchair.configs.compiler_config import CompilerConfig
        import torchair as tng
        config = CompilerConfig()
        config.debug.graph_dump.type = "pbtxt"
        npu_backend = tng.get_npu_backend(compiler_config=config)
        class NPUQuantizedLinearA16W8(torch.nn.Module):
            def __init__(self,
                         weight,
                         antiquant_scale,
                         antiquant_offset,
                         quant_offset=None,
                         quant_scale=None,
                         bias=None,
                         transpose_x=False,
                         transpose_weight=True,
                         w4=False):
                super().__init__()
        
                self.dtype = torch.float16
                self.weight = weight.to(torch.int8).npu()
                self.transpose_weight = transpose_weight
        
                if self.transpose_weight:
                    self.weight = torch_npu.npu_format_cast(self.weight.contiguous(), 29)
                else:
                    self.weight = torch_npu.npu_format_cast(self.weight.transpose(0, 1).contiguous(), 29) # n,k ->nz
        
                self.bias = None
                self.antiquant_scale = antiquant_scale
                self.antiquant_offset = antiquant_offset
                self.quant_offset = quant_offset
                self.quant_scale = quant_scale
                self.transpose_x = transpose_x
        
            def forward(self, x):
                x = torch_npu.npu_weight_quant_batchmatmul(x.transpose(0, 1) if self.transpose_x else x,
                                                           self.weight.transpose(0, 1),
                                                           self.antiquant_scale.transpose(0, 1),
                                                           self.antiquant_offset.transpose(0, 1),
                                                           self.quant_scale,
                                                           self.quant_offset,
                                                           self.bias)
                return x
        
        
        m, k, n = 4, 1024, 4096
        cpu_x = torch.randn((m, k),dtype=torch.float16)
        cpu_weight = torch.randint(1, 10, (k, n),dtype=torch.int8)
        cpu_weight = cpu_weight.transpose(-1, -2)
        
        cpu_antiquantscale = torch.randn((1, n),dtype=torch.float16)
        cpu_antiquantoffset = torch.randn((1, n),dtype=torch.float16)
        cpu_antiquantscale = cpu_antiquantscale.transpose(-1, -2)
        cpu_antiquantoffset = cpu_antiquantoffset.transpose(-1, -2)
        model = NPUQuantizedLinearA16W8(cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu())
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        out = model(cpu_x.npu())
        ```
